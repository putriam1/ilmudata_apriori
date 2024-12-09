from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import os
import uuid

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secure key

# Folder paths
UPLOAD_FOLDER = 'static/uploads/'
PROCESSED_FOLDER = 'static/uploads/processed/'

# Ensure processed folder exists
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        missing_data = request.form.get('missing_data')

        if file and file.filename.endswith('.csv'):
            # Generate a unique filename to prevent overwriting
            filename = f"{UPLOAD_FOLDER}{uuid.uuid4()}_{file.filename}"
            file.save(filename)
            
            # Read the CSV file
            try:
                df = pd.read_csv(filename)
            except Exception as e:
                flash(f"Error reading the CSV file: {e}")
                return redirect(url_for('index'))

            # Handle missing data
            if missing_data == 'drop':
                df = df.dropna()
            elif missing_data == 'fill_0':
                df = df.fillna(0)
            elif missing_data == 'fill_mode':
                df = df.apply(lambda x: x.fillna(x.mode()[0]), axis=0)

            # Create a column 'items' that contains a list of items
            df['items'] = df.apply(lambda row: row.dropna().tolist(), axis=1)

            # Save processed data in the 'processed' folder
            processed_filename = f"{PROCESSED_FOLDER}{uuid.uuid4()}_processed_{file.filename}"
            df.to_csv(processed_filename, index=False)

            # Pass processed data and filename to the template
            return render_template('index.html', preprocessed_data=df.to_html(classes='table table-bordered'), processed_filename=processed_filename)

        else:
            flash("Please upload a valid CSV file.")
            return redirect(url_for('index'))

    return render_template('index.html')

@app.route('/apriori_analysis', methods=['POST'])
def apriori_analysis():
    # Retrieve parameters from the form
    min_support_str = request.form.get('min_support')
    min_confidence_str = request.form.get('min_confidence')
    k_value = request.form.get('k_value')  # This should be validated to ensure it's a valid integer
    specific_item_str = request.form.get('specific_item', '[]')  # Default to an empty list if not provided

    # Ensure min_support and min_confidence are valid floats
    if not min_support_str or not min_confidence_str:
        flash("Please fill in both Minimum Support and Minimum Confidence.")
        return redirect(url_for('index'))

    try:
        min_support = float(min_support_str)
        min_confidence = float(min_confidence_str)
    except ValueError:
        flash("Minimum Support and Minimum Confidence must be valid numbers.")
        return redirect(url_for('index'))

    # Ensure k_value is a valid integer
    try:
        k_value = int(k_value)
    except ValueError:
        flash("Frequent Itemset K (Jumlah Item) must be a valid integer.")
        return redirect(url_for('index'))

    # Convert the specific_item string to a list
    try:
        specific_item = eval(specific_item_str)  # Convert string to list (default is [])
        if not isinstance(specific_item, list):
            raise ValueError
    except (ValueError, SyntaxError):
        flash("Specific Item must be a valid list (e.g., ['item1', 'item2']).")
        return redirect(url_for('index'))

    processed_filename = request.form.get('processed_filename')

    if not processed_filename:
        flash("Processed file path is missing!")
        return redirect(url_for('index'))

    try:
        # Load preprocessed data
        df = pd.read_csv(processed_filename)
    except Exception as e:
        flash(f"Error loading the processed CSV file: {e}")
        return redirect(url_for('index'))

    # Prepare transactions (items)
    transactions = df['items'].tolist()

    # Convert transactions to one-hot encoded DataFrame
    transactions_encoded = pd.get_dummies(pd.DataFrame(transactions), prefix='', prefix_sep='')

    # Run Apriori algorithm
    frequent_itemsets = apriori(transactions_encoded, min_support=min_support, use_colnames=True)
    apriori_results = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence, num_itemsets=len(frequent_itemsets))

    # Filter results if specific_item is provided
    if specific_item:
        apriori_results = apriori_results[apriori_results['antecedents'].apply(lambda x: all(item in x for item in specific_item))]

    # Convert results to HTML
    apriori_results_html = apriori_results.to_html(classes='table table-bordered')

    return render_template('index.html', apriori_results=apriori_results_html)

if __name__ == '__main__':
    app.run(debug=True)
