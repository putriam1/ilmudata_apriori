from flask import Flask, render_template, request, redirect, url_for, flash
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
        missing_data_method = request.form['missing_data']

        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            try:
                # Baca dataset
                df = pd.read_csv(filepath)

                # Tangani missing value
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

from flask import Flask, render_template, request, redirect, url_for, flash
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
        missing_data_method = request.form['missing_data']

        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            try:
                # Baca dataset
                df = pd.read_csv(filepath)

                # Tangani missing value
                if missing_data_method == 'drop':
                    df.dropna(inplace=True)
                elif missing_data_method == 'fill_0':
                    df.fillna(0, inplace=True)
                elif missing_data_method == 'fill_mode':
                    df.fillna(df.mode().iloc[0], inplace=True)

                # Pisahkan aktivitas menjadi list dalam format ['Item1', 'Item2']
                if 'Aktivitas' in df.columns:
                    df['Aktivitas'] = df['Aktivitas'].apply(
                        lambda x: str([item.strip() for item in x.split(',')]) if isinstance(x, str) else "[]"
                    )

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

@app.route('/apriori-analysis', methods=['POST'])
def apriori_analysis():
    try:
        min_support = float(request.form['min_support'])
        min_confidence = float(request.form['min_confidence'])
        k_value = int(request.form['k_value'])

        specific_item = ast.literal_eval(request.form['specific_item'])
        processed_filename = request.form['processed_filename']
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)

        # Validasi data file
        if not os.path.exists(processed_path):
            flash(f'File tidak ditemukan: {processed_filename}', 'danger')
            return redirect(url_for('upload_and_preprocess'))

        df = pd.read_csv(processed_path)

        # Konversi ke transaksi (drop NaN dan buat list per transaksi)
        transactions = df.apply(lambda row: row.dropna().tolist(), axis=1).tolist()

        # One-hot encoding
        encoded_data = pd.get_dummies(pd.DataFrame(transactions).stack()).groupby(level=0).sum().astype(bool)

        # Apriori
        frequent_itemsets = apriori(encoded_data, min_support=min_support, use_colnames=True)

        # Temukan aturan asosiasi
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_confidence, support_only=True, num_itemsets=len(frequent_itemsets))

        # Filter aturan asosiasi berdasarkan item spesifik yang diminta
        if specific_item:
            rules = rules[rules['antecedents'].apply(lambda x: specific_item.issubset(x))]

        # Simpan aturan asosiasi
        rules_filename = f"rules_{time.time()}.csv"
        rules_filepath = os.path.join(app.config['PROCESSED_FOLDER'], rules_filename)
        rules.to_csv(rules_filepath, index=False)

        flash(f'Analisis Apriori selesai! Aturan asosiasi disimpan di {rules_filename}', 'success')

        return render_template('results.html', rules=rules.to_html(classes='table table-bordered'), rules_filename=rules_filename)

    except Exception as e:
        flash(f'Error: {str(e)}', 'danger')
        return redirect(url_for('upload_and_preprocess'))

if __name__ == '__main__':
    app.run(debug=True)

