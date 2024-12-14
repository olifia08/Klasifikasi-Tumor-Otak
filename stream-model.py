import pickle
import streamlit as st
import numpy as np
import pandas as pd
import sklearn.model_selection as ms
import os  # Untuk validasi file
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# Path absolut ke file model dan dataset
# model_path = r"D:\UTM Teknik Informatika\Semester 5\PSD\UAS\PSD_Olif\Deploy\model_logistikregression.sav"
# dataset_path = r"D:\UTM Teknik Informatika\Semester 5\PSD\UAS\PSD_Olif\Deploy\TCGA_InfoWithGrade.csv"
model_path = os.path.join("model_logistikregression.sav")
dataset_path = os.path.join("TCGA_InfoWithGrade.csv")

# Membaca model
try:
    model_glioma = pickle.load(open(model_path, 'rb'))
except FileNotFoundError:
    st.error("File model tidak ditemukan. Pastikan path file benar.")



# Sidebar menu
menu = st.sidebar.selectbox(
    "Main Menu",
    options=["Home", "Klasifikasi"],
    index=0
)

# Membaca dataset jika tersedia
if os.path.exists(dataset_path):
    try:
        cgga_df = pd.read_csv(dataset_path)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca dataset: {e}")
else:
    cgga_df = None
    st.warning("Dataset tidak ditemukan. Pastikan file dataset tersedia.")

# Tampilan Menu "Home" (Dokumentasi)
if menu == "Home":
    st.title("Klasifikasi penyakit Glioma (Tumor Otak)")
    st.markdown(
        """
        Klasifikasi penyakit Glioma (Tumor Otak) untuk membedakan **LGG** (Glioma Tingkat Rendah) dengan inisialisasi 0 dan **GBM** 
        (Glioblastoma Multiforme) di inisialisasikan 1. penelitian ini menggunakan Metode utama Logistik Regression yang nantinya dibandingkan dengan dua metode lainya yaitu Random Forest, dan K-NN.

        **Data yang digunakan:**
        Dataset Glioma yang diperoleh melalui website UCI dengan total **839 data** dan **23 fitur** (1 fitur sebagai kelas).
        Dari 839 data, terdapat:
        - 352 data Glioblastoma Multiforme (GBM)
        - 487 data Lower Grade Glioma (LGG)
        """
    )
    
    if cgga_df is not None:
        st.markdown("## Sampel Dataset")
        st.dataframe(cgga_df)  # Menampilkan sampel data
        st.markdown("## Informasi Dataset")
        st.dataframe(cgga_df.describe())
        
        # Path gambar
        chart_age_path = "chart_age.png"
        chart_grade_path = "chart_grade.png"
    
        # Tampilkan gambar jika ada
        st.markdown("### Penyebaran Data")
    
        # Gambar distribusi usia
        if os.path.exists(chart_age_path):
            try:
                st.image(chart_age_path, caption="Distribusi Usia", use_container_width=True)  # Ganti use_column_width
            except Exception as e:
                st.warning(f"Gagal memuat gambar distribusi usia: {e}")
        else:
            st.warning("Gambar distribusi usia tidak tersedia.")
    
        # Gambar distribusi grade
        if os.path.exists(chart_grade_path):
            try:
                st.image(chart_grade_path, caption="Distribusi Grade", use_container_width=True)  # Ganti use_column_width
            except Exception as e:
                st.warning(f"Gagal memuat gambar distribusi grade: {e}")
        else:
            st.warning("Gambar distribusi grade tidak tersedia.")
        
        st.markdown("## Preprocessing")
        st.markdown("1. Mendeteksi Missing Value")
        st.dataframe(cgga_df.isna().sum())
        st.markdown("2. Normalisasi Data")
        
        def min_max_normalize(column):
            return (column - column.min()) / (column.max() - column.min())
        
        # Salin dataset untuk normalisasi
        normalisasi = cgga_df.copy()
        
        # Terapkan normalisasi pada kolom 'Age_at_diagnosis'
        if 'Age_at_diagnosis' in normalisasi.columns:
            normalisasi['Age_at_diagnosis'] = min_max_normalize(cgga_df['Age_at_diagnosis'])
            st.dataframe(normalisasi)
        else:
            st.warning("Kolom 'Age_at_diagnosis' tidak ditemukan dalam dataset.")
    
        # Penanganan outlier
        st.markdown("3. Outlier")
        x = normalisasi.drop('Grade', axis=1, errors='ignore')  # Pastikan Grade ada
        numerical_features = x.select_dtypes(include=['int64', 'float64'])
    
        # Terapkan LOF hanya pada fitur numerik
        lof = LocalOutlierFactor(n_neighbors=10, contamination=0.15)
        y_outlier = lof.fit_predict(numerical_features)
    
        # Skor anomali
        anomaly_scores = lof.negative_outlier_factor_
    
        # Tambahkan hasil prediksi dan skor ke dataset
        normalisasi['LOF_Prediksi'] = y_outlier
        normalisasi['LOF_Skor_Anomali'] = anomaly_scores
    
        # Hapus data outlier (LOF_Prediksi = -1)
        data_bersih = normalisasi[normalisasi['LOF_Prediksi'] != -1]
        data_bersih = data_bersih.drop(['LOF_Prediksi', 'LOF_Skor_Anomali'], axis=1)
    
        # Visualisasi Outlier dan Data Bersih
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
        # Grafik Outlier
        ax1.scatter(np.arange(len(anomaly_scores)), anomaly_scores, c=y_outlier, cmap='coolwarm', marker='o')
        ax1.axhline(0, color='black', linestyle='--')
        ax1.set_title('Grafik Outlier (LOF)')
        ax1.set_xlabel('Index Data')
        ax1.set_ylabel('Skor Anomali')
    
        # Jumlah Data Bersih
        data_bersih_count = data_bersih.shape[0]
        ax2.pie([data_bersih_count, len(normalisasi) - data_bersih_count], labels=['Data Bersih', 'Outlier'], autopct='%1.1f%%', startangle=90)
        ax2.set_title('Distribusi Data Bersih vs Outlier')
    
        # Tampilkan Grafik di Streamlit
        st.pyplot(fig)
        st.dataframe(data_bersih)
    
        # Split Data
        X = data_bersih[['Age_at_diagnosis', 'Gender', 'Race', 'IDH1', 'TP53', 'ATRX', 'PTEN', 'EGFR',
                         'CIC', 'MUC16', 'PIK3CA', 'NF1', 'PIK3R1', 'FUBP1', 'RB1', 'NOTCH1', 'BCOR', 'CSMD3',
                         'SMARCA4', 'GRIN2A', 'IDH2', 'FAT4', 'PDGFRA']]
        y = data_bersih['Grade']
        X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.20, random_state=0)
    
    
    
        st.markdown("## Modelling Logistik Regression")
    
        def sigmoid(z):
            return 1 / (1 + np.exp(-z))
    
        def predict(X, weights):
            return sigmoid(np.dot(X, weights))
    
        def predict_labels(X, weights, threshold=0.5):
            probabilities = predict(X, weights)
            return (probabilities >= threshold).astype(int)
    
        def update_weights(X, y, weights, learning_rate):
             for i in range(len(y)):
                X_i = X[i].reshape(-1)
                y_pred = predict(X_i, weights)
                error = y_pred - y[i]
                weights -= learning_rate * error * X_i
                return weights
    
        def logistic_regression_sgd(X, y, learning_rate, epochs):
            np.random.seed(42)
            weights = np.random.rand(X.shape[1]) * 0.01
            for epoch in range(epochs):
                weights = update_weights(X, y, weights, learning_rate)
                accuracy = calculate_accuracy(X, y, weights)
                return weights
    
        def calculate_accuracy(X, y, weights):
            y_pred = predict_labels(X, weights)
            accuracy = np.mean(y_pred == y)
            return accuracy
    
        # Split data
        X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
        X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]
        y_train = np.ravel(np.array(y_train))
        y_test = np.ravel(np.array(y_test))
        
        # Pelatihan model
        learning_rate = 0.01
        epochs = 50
        weights = logistic_regression_sgd(X_train, y_train, learning_rate, epochs)
    
        # Prediksi pada data uji
        y_pred_test = predict_labels(X_test, weights)
        conf_matrix = confusion_matrix(y_test, y_pred_test)
    
        # Akurasi model pada data pelatihan
        train_accuracy = calculate_accuracy(X_train, y_train, weights)
    
        # Mengganti label numerik dengan label teks
        label_mapping = {0: "LGG", 1: "GBM"}
        y_test_mapped = pd.Series(y_test).map(label_mapping)
        y_pred_test_mapped = pd.Series(y_pred_test).map(label_mapping)
    
        # Akurasi 
        st.markdown(f"### Akurasi Model: {train_accuracy * 100:.2f}%")
        # Menampilkan 10 sampel data klasifikasi dalam tabel
        st.markdown("### Sampel Data Klasifikasi")
        classification_results = pd.DataFrame({
            'True Label': y_test_mapped,               
            'Predicted Label': y_pred_test_mapped
        })
        sample_results = classification_results.sample(10)
        st.dataframe(sample_results) # Menampilkan tabel hasil 10 sampel

        # Classification report
        st.markdown("### Classification Report")
        report = classification_report(y_test, y_pred_test, target_names=['LGG', 'GBM'], output_dict=True)
        classification_df = pd.DataFrame(report).transpose()
        st.dataframe(classification_df)
    
        # Plot confusion matrix
        st.markdown("### Confusion Matrix")
        plt_lg = plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['LGG', 'GBM'], yticklabels=['LGG', 'GBM'])
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        st.pyplot(plt_lg)
    
    
    
    
    
        # Modelling Random Forest
        st.markdown("## Modelling Random Forest")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            max_features=10
        )
        model.fit(X_train, y_train)
        y_pred_randomforest = model.predict(X_test)
        score_randomforest = model.score(X_test, y_test)
        conf_matrix_randomforest = confusion_matrix(y_test, y_pred_randomforest)
        # Ganti label 0 dengan "LGG" dan 1 dengan "GBM"
        y_test_mapped = pd.Series(y_test).map(label_mapping)
        y_pred_randomforest_mapped = pd.Series(y_pred_randomforest).map(label_mapping)    
            # Akurasi
        st.markdown(f"### Akurasi Model: {score_randomforest * 100:.2f}%")
        # Menampilkan 10 sampel data klasifikasi dalam tabel
        classification_randomforest_results = pd.DataFrame({                
            'True Label': y_test_mapped,
            'Predicted Label': y_pred_randomforest_mapped
        })
        sample_randomforest_results = classification_randomforest_results.sample(10)
        st.markdown("### Sampel Hasil Prediksi")
        st.dataframe(sample_randomforest_results)
    
        # Classification report
        report_randomforest = classification_report(y_test, y_pred_randomforest, target_names=['LGG', 'GBM'], output_dict=True)            
        classification_randomforest_df = pd.DataFrame(report_randomforest).transpose()
        st.markdown("### Classification Report")
        st.dataframe(classification_randomforest_df)    
        # Plot confusion matrix
        st.markdown("### Confusion Matrix")
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix_randomforest, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['LGG', 'GBM'], yticklabels=['LGG', 'GBM'])
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix (Random Forest)')
        st.pyplot(plt)
    
    
    
    
    
        st.markdown("## Modelling K-NN")
    
        try:
            # Inisialisasi model K-NN
            model_knn = KNeighborsClassifier(n_neighbors=10)
            model_knn.fit(X_train, y_train)
    
            # Prediksi dan evaluasi
            y_pred_knn = model_knn.predict(X_test)
            score_knn = accuracy_score(y_test, y_pred_knn)
            conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
    
            # Ganti label 0 dengan "LGG" dan 1 dengan "GBM"
            label_mapping = {0: "LGG", 1: "GBM"}
            y_test_mapped = pd.Series(y_test).map(label_mapping)
            y_pred_knn_mapped = pd.Series(y_pred_knn).map(label_mapping)
    
            # Akurasi K-NN
            st.markdown(f"### Akurasi Model: {score_knn * 100:.2f}%")
            # Menampilkan 10 sampel data klasifikasi dalam tabel
            st.markdown("#### Sampel Klasifikasi 10 Data")
            classification_knn_results = pd.DataFrame({
                'True Label': y_test_mapped,                    
                'Predicted Label': y_pred_knn_mapped
            })
    
            # Penyesuaian jumlah sampel untuk dataset kecil
            if len(classification_knn_results) > 0:
                sample_knn_results = classification_knn_results.sample(min(10, len(classification_knn_results)))                    
                st.dataframe(sample_knn_results)  # Menampilkan tabel hasil 10 sampel
            else:
                st.warning("Tidak ada data yang cukup untuk menampilkan sampel klasifikasi.")
    
            # Classification report
            st.markdown("#### Classification Report")
            report_knn = classification_report(y_test, y_pred_knn, target_names=['LGG', 'GBM'], output_dict=True)
            classification_knn_df = pd.DataFrame(report_knn).transpose()
            st.dataframe(classification_knn_df)  # Menampilkan classification report dalam bentuk tabel

            # Plot confusion matrix
            st.markdown("#### Confusion Matrix (KNN)")
            plt_knn = plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap='Blues', xticklabels=['LGG', 'GBM'], yticklabels=['LGG', 'GBM'])
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix (KNN)')
            st.pyplot(plt_knn)
    
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
    
    
        st.markdown("## Perbandingan 3 Metode")
        
         # Membuat DataFrame untuk menyimpan hasil model
        models = pd.DataFrame({
            'Model': ['Logistic Regression', 'Random Forest Classifier', 'KNN Classifier'],               
            'Score': [train_accuracy, score_randomforest, score_knn]
         })
         # Mengurutkan berdasarkan nilai akurasi dari yang tertinggi ke terendah
        models_sorted = models.sort_values(by='Score', ascending=False)
        # Plot perbandingan skor menggunakan Seaborn
        st.markdown("### Grafik Perbandingan Akurasi")
        plt.figure(figsize=(8, 6))
        sns.barplot(x='Score', y='Model', data=models_sorted, palette='viridis')
        plt.xlabel('Accuracy Score')
        plt.ylabel('Model')
        plt.title('Comparison of Model Accuracy')
        st.pyplot(plt)

        if 'LGG' in classification_df.index and 'GBM' in classification_df.index:
            precision_avg = (classification_df.loc['LGG', 'precision'] + classification_df.loc['GBM', 'precision']) / 2
            recall_avg = (classification_df.loc['LGG', 'recall'] + classification_df.loc['GBM', 'recall']) / 2
            f1_score_avg = (classification_df.loc['LGG', 'f1-score'] + classification_df.loc['GBM', 'f1-score']) / 2
        else:
            st.error("Label LGG atau GBM tidak ditemukan di classification_df.")
            precision_avg, recall_avg, f1_score_avg = 0, 0, 0
        
            # Simpan hasil ke DataFrame perbandingan
        metrics_df = pd.DataFrame({
            'Model': ['Logistic Regression', 'Random Forest', 'KNN'],
            'Precision': [
                precision_avg,
                (report_randomforest['LGG']['precision'] + report_randomforest['GBM']['precision']) / 2,
                (report_knn['LGG']['precision'] + report_knn['GBM']['precision']) / 2
            ],
            'Recall': [
                recall_avg,
                (report_randomforest['LGG']['recall'] + report_randomforest['GBM']['recall']) / 2,
                (report_knn['LGG']['recall'] + report_knn['GBM']['recall']) / 2
            ],
            'F1-Score': [
                f1_score_avg,
                (report_randomforest['LGG']['f1-score'] + report_randomforest['GBM']['f1-score']) / 2,
                (report_knn['LGG']['f1-score'] + report_knn['GBM']['f1-score']) / 2
            ]
        })    
    
         # Plot bar chart untuk perbandingan
        st.markdown("### Grafik Perbandingan Precision, Recall, dan F1-Score")
        metrics_melted = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Metric', y='Score', hue='Model', data=metrics_melted, palette='viridis')
        plt.title('Comparison of Precision, Recall, and F1-Score')
        plt.ylabel('Score')
        # plt.ylim(0, 1)  # Batas untuk nilai metrik          
        plt.xlabel('Metric')
        plt.legend(title='Model')
        st.pyplot(plt)
    
    
    else:
        st.error("Dataset tidak tersedia untuk ditampilkan.")
# Tampilan Menu "Klasifikasi"
elif menu == "Klasifikasi":
    st.title("Klasifikasi Glioma")
    st.markdown("### Masukkan Data Pasien:")

    # Form input data klasifikasi
    Age_at_diagnosis = st.number_input("Umur", min_value=0)
    Gender = st.selectbox("Jenis Kelamin", options=["P", "L"])
    Race = st.selectbox("Suku", options=["White", "Black or African", "American", "Asian", "American India or Alaska Native"])
    IDH1 = st.selectbox("Status IDH1", options=["Tidak Mutasi", "Mutasi"])
    TP53 = st.selectbox("Status TP53", options=["Tidak Mutasi", "Mutasi"])
    ATRX = st.selectbox("Status ATRX", options=["Tidak Mutasi", "Mutasi"])
    PTEN = st.selectbox("Status PTEN", options=["Tidak Mutasi", "Mutasi"])
    EFGR = st.selectbox("Status EFGR", options=["Tidak Mutasi", "Mutasi"])
    CIC = st.selectbox("Status CIC", options=["Tidak Mutasi", "Mutasi"])
    MUC16 = st.selectbox("Status MUC16", options=["Tidak Mutasi", "Mutasi"])
    PIK3CA = st.selectbox("Status PIK3CA", options=["Tidak Mutasi", "Mutasi"])
    NF1 = st.selectbox("Status NF1", options=["Tidak Mutasi", "Mutasi"])
    PIK3R1 = st.selectbox("Status PIK3R1", options=["Tidak Mutasi", "Mutasi"])
    FUBP1 = st.selectbox("Status FUBP1", options=["Tidak Mutasi", "Mutasi"])
    RB1 = st.selectbox("Status RB1", options=["Tidak Mutasi", "Mutasi"])
    NOTCH1 = st.selectbox("Status NOTCH1", options=["Tidak Mutasi", "Mutasi"])
    BCOR = st.selectbox("Status BCOR", options=["Tidak Mutasi", "Mutasi"])
    CSMD3 = st.selectbox("Status CSMD3", options=["Tidak Mutasi", "Mutasi"])
    SMARCA4 = st.selectbox("Status SMARCA4", options=["Tidak Mutasi", "Mutasi"])
    GRIN2A = st.selectbox("Status GRIN2A", options=["Tidak Mutasi", "Mutasi"])
    IDH2 = st.selectbox("Status IDH2", options=["Tidak Mutasi", "Mutasi"])
    FAT4 = st.selectbox("Status FAT4", options=["Tidak Mutasi", "Mutasi"])
    PDGFRA = st.selectbox("Status PDGFRA", options=["Tidak Mutasi", "Mutasi"])

    # Tombol Prediksi
    if st.button("Klasifikasi"):
        # Transformasi input ke dalam format yang sesuai untuk model
        input_data = np.array([
            1,  # Bias tambahan
            Age_at_diagnosis,
            1 if Gender == "P" else 0,  # Gender encoding
            0 if Race == "White" else 1 if Race == "Black or African" else 2 if Race == "Asian" else 3,
            1 if IDH1 == "Mutasi" else 0,
            1 if TP53 == "Mutasi" else 0,
            1 if ATRX == "Mutasi" else 0,
            1 if PTEN == "Mutasi" else 0,
            1 if EFGR == "Mutasi" else 0,
            1 if CIC == "Mutasi" else 0,
            1 if MUC16 == "Mutasi" else 0,
            1 if PIK3CA == "Mutasi" else 0,
            1 if NF1 == "Mutasi" else 0,
            1 if PIK3R1 == "Mutasi" else 0,
            1 if FUBP1 == "Mutasi" else 0,
            1 if RB1 == "Mutasi" else 0,
            1 if NOTCH1 == "Mutasi" else 0,
            1 if BCOR == "Mutasi" else 0,
            1 if CSMD3 == "Mutasi" else 0,
            1 if SMARCA4 == "Mutasi" else 0,
            1 if GRIN2A == "Mutasi" else 0,
            1 if IDH2 == "Mutasi" else 0,
            1 if FAT4 == "Mutasi" else 0,
            1 if PDGFRA == "Mutasi" else 0
        ])

        # Lakukan prediksi
        try:
            # Memuat model
            with open(model_path, 'rb') as file:
                weights = pickle.load(file)

            # Gunakan fungsi predict_labels untuk prediksi
            def predict_labels(X, weights, threshold=0.5):
                probabilities = 1 / (1 + np.exp(-np.dot(X, weights)))
                return (probabilities >= threshold).astype(int)

            hasil_prediksi = predict_labels(input_data.reshape(1, -1), weights)[0]

            # Mapping hasil prediksi ke label
            label_mapping = {0: "LGG (Glioma Tingkat Rendah)", 1: "GBM (Glioblastoma Multiforme)"}
            st.success(f"Hasil Prediksi: {label_mapping[hasil_prediksi]}")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
