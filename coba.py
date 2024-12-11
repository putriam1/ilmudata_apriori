from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import os
import ast
import time

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = os.path.join(UPLOAD_FOLDER, 'processed')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Buat folder untuk upload dan processed jika belum ada
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

@app.route('/', methods=['GET', 'POST'])
def upload_and_preprocess():
    preprocessed_data = None
    processed_filename = None

    if request.method == 'POST':
        # Ambil file CSV dari form
        file = request.files['file']
        missing_data_method = request.form.get('missing_data', None)  # Gunakan get() untuk mencegah KeyError

        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            try:
                # Baca dataset
                df = pd.read_csv(filepath)

                # Tangani missing value jika 'missing_data' ada
                if missing_data_method:
                    if missing_data_method == 'drop':
                        df.dropna(inplace=True)
                    elif missing_data_method == 'fill_0':
                        df.fillna(0, inplace=True)
                    elif missing_data_method == 'fill_mode':
                        df.fillna(df.mode().iloc[0], inplace=True)

                # Nama file hasil pra-pemrosesan
                processed_filename = f"processed_{file.filename}"
                processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)

                # Simpan hasil pra-pemrosesan ke folder yang benar
                df.to_csv(processed_path, index=False)

                # Verifikasi apakah file berhasil disimpan
                if os.path.exists(processed_path):
                    flash(f'File berhasil diproses dan disimpan sebagai {processed_filename}', 'success')
                else:
                    flash('Terjadi masalah saat menyimpan file yang diproses.', 'danger')

                # Tampilkan data yang telah diproses
                preprocessed_data = df.to_html(classes='table table-striped table-bordered', justify='center')

            except Exception as e:
                flash(f'Error saat membaca atau memproses file: {e}', 'danger')
                return redirect(url_for('upload_and_preprocess'))
        else:
            flash('Harap unggah file CSV yang valid!', 'danger')

    return render_template('index.html', preprocessed_data=preprocessed_data, processed_filename=processed_filename)


@app.route('/apriori', methods=['GET'])
def apriori_page():
    processed_filename = request.args.get('processed_filename')
    return render_template('apriori.html', processed_filename=processed_filename)

@app.route('/apriori-analysis', methods=['POST'])
def apriori_analysis():
    try:
        min_support = float(request.form['min_support'])
        min_confidence = float(request.form['min_confidence'])
        lhs_length = int(request.form['lhs_length'])
        search_item = request.form.get('search_item', None)
        processed_filename = request.form['processed_filename']
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)

        # Membaca dataset
        data = pd.read_csv(processed_path)
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

        return render_template('results.html', frequent_itemsets_html=frequent_itemsets_html, rules_html=rules_html)

    except KeyError as e:
        return jsonify({"error": f"KeyError: {str(e)} - Periksa apakah semua parameter yang dibutuhkan sudah dikirimkan."})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
