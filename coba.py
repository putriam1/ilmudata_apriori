from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Buat folder untuk upload jika belum ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def upload_and_preprocess():
    preprocessed_data = None
    processed_filename = None

    if request.method == 'POST':
        # Ambil file CSV dari form
        file = request.files['file']
        missing_data_method = request.form['missing_data']

        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Baca dataset
            df = pd.read_csv(filepath)

            # Tangani missing value
            if missing_data_method == 'drop':
                df.dropna(inplace=True)
            elif missing_data_method == 'fill_0':
                df.fillna(0, inplace=True)
            elif missing_data_method == 'fill_mode':
                df.fillna(df.mode().iloc[0], inplace=True)

            # Membuat list untuk setiap baris
            transactions = df.apply(lambda row: row.dropna().tolist(), axis=1).tolist()

            # Simpan hasil pra-pemrosesan
            processed_filename = f"processed_{file.filename}"
            processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
            df.to_csv(processed_path, index=False)

            # Tampilkan data transaksi dalam format list
            transactions_display = [', '.join(map(str, transaction)) for transaction in transactions]
            preprocessed_data = df.to_html(classes='table table-striped table-bordered', justify='center')
            return render_template('index.html', preprocessed_data=preprocessed_data, processed_filename=processed_filename, transactions_display=transactions_display)

            flash('File berhasil diunggah dan diproses!', 'success')
        else:
            flash('Harap unggah file CSV yang valid!', 'danger')

    return render_template('index.html', preprocessed_data=preprocessed_data, processed_filename=processed_filename)

@app.route('/apriori-analysis', methods=['POST'])
def apriori_analysis():
    min_support = float(request.form['min_support'])
    min_confidence = float(request.form['min_confidence'])
    k_value = int(request.form['k_value'])
    specific_item = eval(request.form['specific_item'])
    processed_filename = request.form['processed_filename']

    # Baca file yang sudah diproses
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
    df = pd.read_csv(filepath)

    # Gabungkan semua kolom gejala menjadi transaksi
    transactions = df.apply(lambda row: row.dropna().tolist(), axis=1).tolist()
    
    # Konversi transaksi ke format one-hot encoding
    encoded_data = pd.get_dummies(pd.DataFrame(transactions).stack()).groupby(level=0).sum()

    # Jalankan algoritma apriori
    frequent_itemsets = apriori(encoded_data, min_support=min_support, use_colnames=True)

    # Pastikan itemsets yang memenuhi k_value
    frequent_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) == k_value)]

    # Filter berdasarkan item tertentu (gunakan set untuk mencocokkan item)
    frequent_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: set(specific_item).issubset(x))]

    # Buat aturan asosiasi dengan support_only=True jika masalah 'frozenset' muncul
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence, support_only=True, num_itemsets=len(frequent_itemsets))

    apriori_results = rules.to_html(classes='table table-striped table-bordered', justify='center')

    return render_template('index.html', apriori_results=apriori_results)

if __name__ == '__main__':
    app.run(debug=True)