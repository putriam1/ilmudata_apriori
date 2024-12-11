from flask import Flask, request, render_template, jsonify
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('abc.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        # Ambil data dari form
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "Tidak ada file yang diunggah."})

        min_support = float(request.form['min_support'])
        min_confidence = float(request.form['min_confidence'])
        lhs_length = int(request.form['lhs_length'])
        search_item = request.form.get('specific_item', None)

        # Membaca dataset
        data = pd.read_csv(file)
        data['Aktivitas'] = data['Aktivitas'].str.split(', ')
        transactions = data['Aktivitas']

        # Konversi data transaksi ke dalam bentuk one-hot encoding
        one_hot = pd.get_dummies(transactions.apply(pd.Series).stack()).groupby(level=0).sum()
        one_hot = one_hot.astype(bool)

        # Menerapkan algoritma Apriori
        frequent_itemsets = apriori(one_hot, min_support=min_support, use_colnames=True)

        if frequent_itemsets.empty:
            return jsonify({"error": "Tidak ada itemset yang sering ditemukan. Coba turunkan nilai min_support."})

        # Menghasilkan aturan asosiasi
        num_itemsets = frequent_itemsets['support'].count()  # Hitung jumlah itemset
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence, num_itemsets=num_itemsets)

        if rules.empty:
            return jsonify({"error": "Tidak ada aturan asosiasi yang ditemukan. Coba turunkan nilai min_confidence."})

        # Filter aturan berdasarkan panjang LHS
        rules = rules[rules['antecedents'].apply(lambda x: len(x) == lhs_length)]

        if rules.empty:
            return jsonify({"error": f"Tidak ada aturan asosiasi dengan panjang LHS {lhs_length}."})

        # Filter aturan berdasarkan search_item jika diberikan
        if search_item:
            rules = rules[rules['antecedents'].apply(lambda x: search_item in list(x))]

        # Konversi hasil ke dalam tabel HTML
        frequent_itemsets_html = frequent_itemsets.to_html(classes='table table-bordered')
        rules_html = rules.to_html(classes='table table-bordered')

        # Mengembalikan hasil
        return f"<h3>Frequent Itemsets</h3>{frequent_itemsets_html}<h3>Association Rules</h3>{rules_html}"

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)