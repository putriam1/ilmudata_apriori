from flask import Flask, request, render_template, jsonify
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        # Retrieve form data
        file = request.files['file']
        min_support = float(request.form['min_support'])
        min_confidence = float(request.form['min_confidence'])
        k_value = int(request.form['k_value'])
        search_item = request.form.get('search_item', None)

        # Load dataset
        data = pd.read_csv(file)
        data['Aktivitas'] = data['Aktivitas'].str.split(', ')
        transactions = data['Aktivitas']

        # Convert transactional data to one-hot encoding
        one_hot = pd.get_dummies(transactions.apply(pd.Series).stack()).groupby(level=0).sum()
        one_hot = one_hot.astype(bool)

        # Apply Apriori algorithm
        frequent_itemsets = apriori(one_hot, min_support=min_support, use_colnames=True, max_len=k_value)

        if frequent_itemsets.empty:
            return jsonify({"error": "No frequent itemsets found. Try lowering the min_support value."})

        # Generate association rules
        num_itemsets = frequent_itemsets['support'].count()  # Hitung jumlah itemset
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence, num_itemsets=num_itemsets)
# rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

        if rules.empty:
            return jsonify({"error": "No association rules found. Try lowering the min_confidence value."})

        # Filter rules by search_item if provided
        if search_item:
            rules = rules[rules['antecedents'].apply(lambda x: search_item in list(x))]

        # Convert results to HTML tables
        frequent_itemsets_html = frequent_itemsets.to_html(classes='table table-bordered')
        rules_html = rules.to_html(classes='table table-bordered')

        # Return results
        return f"<h3>Frequent Itemsets</h3>{frequent_itemsets_html}<h3>Association Rules</h3>{rules_html}"

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
