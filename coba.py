from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

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
            missing_data_option = request.form.get("missing_data", "drop")

            if not file:
                flash("Harap unggah file CSV!", "error")
                return redirect(url_for('index'))

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            try:
                data = pd.read_csv(filepath, encoding='ISO-8859-1')

                if data.empty:
                    flash("File CSV kosong. Harap unggah file yang valid.", "error")
                    return redirect(url_for('index'))

                data = data.drop(columns=['Unnamed: 0', 'Unnamed: 1'], errors='ignore')
                data.columns = data.columns.str.replace(';', '')
                data['items'] = data['Item'].apply(lambda x: str(x).split(';'))

                if missing_data_option == "drop":
                    data = data.dropna()
                elif missing_data_option == "fill_0":
                    data = data.fillna(0)
                elif missing_data_option == "fill_mode":
                    data = data.fillna(data.mode().iloc[0])

                preprocessed_data = data.head().to_html(classes='table table-striped')
                session['preprocessed_data'] = data.to_dict()  # Simpan sebagai dictionary

            except UnicodeDecodeError as e:
                flash(f"Gagal membaca file CSV: {e}", "error")
                return redirect(url_for('index'))

        except Exception as e:
            flash(f"Kesalahan tidak terduga: {e}", "error")

    return render_template("index.html", preprocessed_data=preprocessed_data)

@app.route("/apriori", methods=["POST"])
def apriori_analysis():
    try:
        min_support = float(request.form.get("min_support", 0.5))
        min_confidence = float(request.form.get("min_confidence", 0.5))

        if 'preprocessed_data' not in session:
            flash("Data belum diproses. Harap unggah file terlebih dahulu.", "error")
            return redirect(url_for('index'))

        data_dict = session['preprocessed_data']
        data = pd.DataFrame.from_dict(data_dict)

        trx = data['items'].tolist()
        te = TransactionEncoder()
        te_ary = te.fit(trx).transform(trx)
        transactions = pd.DataFrame(te_ary, columns=te.columns_)

        frequent_itemsets = apriori(transactions, min_support=min_support, use_colnames=True)

        if frequent_itemsets.empty:
            flash("Tidak ada itemset yang memenuhi minimum support.", "warning")
            return redirect(url_for('index'))

        # Fungsi association_rules tidak membutuhkan 'num_itemsets'
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

        if rules.empty:
            flash("Tidak ada aturan asosiasi yang ditemukan.", "warning")
            return redirect(url_for('index'))

        results = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_html(classes='table table-striped', index=False)
        return render_template("index.html", results=results)

    except Exception as e:
        flash(f"Kesalahan saat menjalankan Apriori: {e}", "error")
        return redirect(url_for('index'))
    
if __name__ == "__main__":
    app.run(debug=True)
