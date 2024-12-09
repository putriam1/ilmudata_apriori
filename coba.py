from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
import pandas as pd

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.secret_key = 'supersecretkey'

# Pastikan folder upload ada
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route("/", methods=["GET", "POST"])
def index():
    preprocessed_data = None
    if request.method == "POST":
        try:
            file = request.files.get("file")
            missing_data_option = request.form.get("missing_data", "drop")  # Default: drop

            # Validasi file
            if not file:
                flash("Harap unggah file CSV!", "error")
                return redirect(url_for('index'))

            # Simpan file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Baca file CSV dengan encoding yang tepat
            try:
                data = pd.read_csv(filepath, encoding='ISO-8859-1')  # Gunakan encoding yang sesuai

                if data.empty:
                    flash("File CSV kosong. Harap unggah file yang valid.", "error")
                    return redirect(url_for('index'))

                # Tampilkan nama kolom untuk debugging
                print(f"Nama Kolom yang ada dalam file CSV: {data.columns}")  # Debugging

                # Hapus kolom yang tidak perlu
                data = data.drop(columns=['Unnamed: 0', 'Unnamed: 1'], errors='ignore')  # Menghapus kolom yang tidak perlu

                # Perbaiki nama kolom 'Item;' menjadi 'Item'
                data.columns = data.columns.str.replace(';', '')  # Hapus titik koma di nama kolom

                # Pisahkan nilai yang dipisahkan oleh tanda titik koma dan jadikan list
                data['items'] = data['Item'].apply(lambda x: str(x).split(';'))  # Asumsikan kolom bernama 'Item'

                # Hapus item yang memiliki titik koma di akhir nama
                data['items'] = data['items'].apply(lambda x: [item.strip() for item in x if item.strip() != ''])

                # Pra-pemrosesan data berdasarkan pilihan pengguna
                if missing_data_option == "drop":
                    data = data.dropna(axis=0)  # Menghapus baris yang memiliki missing values
                elif missing_data_option == "fill_0":
                    data = data.fillna(0)  # Mengganti missing values dengan 0
                elif missing_data_option == "fill_mean":
                    data = data.fillna(data.mod())  # Mengganti missing values dengan rata-rata

                # Tampilkan hasil pra-pemrosesan untuk debugging
                preprocessed_data = data.head().to_html(classes='table table-striped')
                print(f"Data setelah pra-pemrosesan:\n{data.head()}")  # Debugging

                # Simpan data yang sudah diproses ke session untuk digunakan pada langkah berikutnya
                session['preprocessed_data'] = data.to_html(classes='table table-striped')

            except UnicodeDecodeError as e:
                flash(f"Gagal membaca file CSV: {e}", "error")
                return redirect(url_for('index'))

        except Exception as e:
            flash(f"Kesalahan tidak terduga: {e}", "error")

    return render_template("index.html", preprocessed_data=preprocessed_data)

from mlxtend.frequent_patterns import apriori, association_rules

@app.route("/apriori", methods=["POST"])
def apriori():
    try:
        # Ambil data yang telah diproses dari session
        if 'preprocessed_data' not in session:
            flash("Data belum diproses. Harap unggah file terlebih dahulu.", "error")
            return redirect(url_for('index'))

        # Data sudah diproses dalam session sebagai DataFrame (langsung gunakan session)
        data = pd.read_html(session['preprocessed_data'])[0]  # Mengambil DataFrame dari HTML
        print(f"Data yang digunakan untuk Apriori:\n{data.head()}")  # Debugging

        # Pastikan data di kolom 'items' adalah list yang valid
        # Ubah string yang terlihat seperti list menjadi list yang valid
        data['items'] = data['Item'].apply(lambda x: str(x).strip('[]').replace("'", "").split(';'))
        print(f"Data setelah memperbaiki kolom 'items':\n{data['items'].head()}")  # Debugging

        # Buat kolom untuk setiap item di setiap transaksi
        transactions = data['items'].apply(lambda x: pd.Series(1, index=x)).fillna(0)
        print(f"Transaksi setelah diubah menjadi dummy variables:\n{transactions.head()}")  # Debugging

        # Jalankan algoritma Apriori
        frequent_itemsets = apriori(transactions, min_support=0.5, use_colnames=True)
        print(f"Frequent Itemsets:\n{frequent_itemsets}")  # Debugging

        if frequent_itemsets.empty:
            flash("Tidak ada itemset yang memenuhi minimum support.", "warning")
            return redirect(url_for('index'))

        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
        print(f"Association Rules:\n{rules}")  # Debugging

        if rules.empty:
            flash("Tidak ada aturan asosiasi yang ditemukan.", "warning")
            return redirect(url_for('index'))

        # Tampilkan hasil Apriori
        results = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_html(classes='table table-striped', index=False)

        return render_template("index.html", results=results)

    except Exception as e:
        flash(f"Terjadi kesalahan saat menjalankan Apriori: {e}", "error")
        print(f"Error: {e}")  # Debugging error yang terjadi
        return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(debug=True)
