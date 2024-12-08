from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Pastikan folder untuk upload file ada
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    if request.method == "POST":
        # Proses file upload
        file = request.files.get("file")
        min_support = float(request.form.get("min_support", 0.5))  # Default 0.5
        min_confidence = float(request.form.get("min_confidence", 0.5))  # Default 0.5
        
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Load data
            data = pd.read_csv(filepath)
            
            # Proses Apriori
            try:
                # Ubah data sesuai format transaksi dan pastikan tipe data boolean
                transactions = pd.get_dummies(data, prefix_sep='_')
                transactions = transactions.astype(bool)  # Pastikan tipe boolean
                frequent_itemsets = apriori(transactions, min_support=min_support, use_colnames=True)
                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
                results = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_html(classes='table table-striped', index=False)
            except Exception as e:
                results = f"<p style='color:red;'>Error: {e}</p>"

    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
