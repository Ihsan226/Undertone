# 🎯 Panduan Penggunaan Website CRISP-DM Image Classifier

## 📌 Overview
Website ini adalah aplikasi Streamlit modern dan interaktif yang mengimplementasikan metodologi CRISP-DM (Cross-Industry Standard Process for Data Mining) lengkap untuk klasifikasi gambar ke dalam 3 kelas: **Black**, **Brown**, dan **White**.

## 🚀 Cara Menjalankan

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan Aplikasi
```bash
streamlit run streamlit_app.py
```

### 3. Akses Website
Aplikasi akan otomatis terbuka di browser atau akses di:
- **Local URL:** http://localhost:8501 atau http://localhost:8502
- **Network URL:** http://[your-ip]:8501

## 📋 Fitur Utama

### 🏠 Home Page
- **Overview CRISP-DM:** Visualisasi 6 tahap metodologi CRISP-DM
- **Quick Stats:** Metrik cepat tentang dataset dan model
- **Project Overview:** Penjelasan lengkap tentang proyek
- **Configuration Display:** Melihat konfigurasi yang sedang digunakan

### 💼 Business Understanding
- **Business Objectives:** Tujuan dan KPI proyek
- **Success Criteria:** Target performa yang ingin dicapai
- **Project Plan:** Rencana pengembangan proyek
- **Use Cases:** Contoh aplikasi dalam dunia nyata

### 🔎 Data Understanding
- **Dataset Analysis:** Analisis statistik dataset
- **Class Distribution:** Visualisasi distribusi kelas
- **Sample Images:** Tampilan contoh gambar dari setiap kelas
- **Interactive Charts:** Grafik interaktif dengan Plotly
- **Complete Dataset Table:** Tabel lengkap dataset

### 🧪 Data Preparation
- **Preprocessing Configuration:** Pengaturan preprocessing gambar
- **Pipeline Steps:** Langkah-langkah persiapan data
- **Train/Val/Test Split:** Pembagian dataset
- **Progress Tracking:** Progress bar untuk proses yang berjalan
- **Artifacts Management:** Manajemen file hasil preprocessing

### 🤖 Modeling
- **SVM Configuration:** Pengaturan model SVM
- **Hyperparameter Tuning:** GridSearchCV untuk optimasi parameter
- **Training Progress:** Progress bar saat training
- **Grid Search Results:** Visualisasi hasil tuning
- **Interactive Visualizations:** Grafik perbandingan parameter
- **Model Status:** Status dan informasi model

### 📈 Evaluation
- **Performance Metrics:** Accuracy, Precision, Recall, F1-Score
- **Confusion Matrix:** Visualisasi confusion matrix
- **Classification Report:** Laporan klasifikasi lengkap
- **Per-Class Performance:** Analisis performa per kelas
- **Misclassifications Analysis:** Analisis kesalahan prediksi
- **Sample Misclassified Images:** Contoh gambar yang salah diklasifikasi

### 🔮 Predict
- **Image Upload:** Upload gambar untuk prediksi
- **Real-time Classification:** Klasifikasi instant
- **Confidence Scores:** Skor kepercayaan untuk setiap kelas
- **Interactive Charts:** Visualisasi distribusi probabilitas
- **Preprocessing Info:** Informasi tentang proses preprocessing

## 🎨 Fitur UI Modern

### Design Elements
- ✨ **Gradient Colors:** Warna gradien modern (#667eea → #764ba2)
- 🎴 **Card-based Layout:** Desain berbasis kartu yang clean
- 📊 **Interactive Charts:** Grafik interaktif dengan Plotly
- 🌈 **Color-coded Metrics:** Metrik dengan kode warna
- ⚡ **Smooth Animations:** Animasi dan transisi yang smooth
- 📱 **Responsive Design:** Desain responsif untuk berbagai ukuran layar

### Interactive Components
- 🔘 **Modern Buttons:** Tombol dengan efek hover dan animasi
- 📈 **Progress Bars:** Progress bar untuk proses yang berjalan
- 🎯 **Expandable Sections:** Section yang dapat dibuka/tutup
- 📊 **Interactive Tables:** Tabel dengan gradient styling
- 🖼️ **Image Gallery:** Galeri gambar dengan tabs
- 🎨 **Custom Sidebar:** Sidebar dengan status pipeline

### Visual Feedback
- ✅ **Success Messages:** Pesan sukses dengan styling menarik
- ⚠️ **Warning Alerts:** Peringatan yang jelas
- ❌ **Error Handling:** Error messages yang informatif
- ℹ️ **Info Boxes:** Kotak informasi dengan styling berbeda
- 🔄 **Loading States:** Indikator loading yang menarik

## 📊 Alur Penggunaan CRISP-DM

### 1️⃣ Business Understanding
- Pahami tujuan proyek
- Review KPI dan success criteria
- Lihat use cases yang relevan

### 2️⃣ Data Understanding
- Klik "Run Data Summary Analysis"
- Lihat distribusi kelas
- Explore sample images dari setiap kelas
- Review dataset table

### 3️⃣ Data Preparation
- Review konfigurasi preprocessing
- Klik "Process & Split Dataset"
- Tunggu hingga proses selesai
- Verifikasi artifacts yang dihasilkan

### 4️⃣ Modeling
- Review konfigurasi SVM
- Klik "Train Model (with Hyperparameter Tuning)"
- Tunggu proses training (bisa beberapa menit)
- Review hasil grid search
- Lihat visualisasi parameter tuning

### 5️⃣ Evaluation
- Klik "Run Evaluation & Generate Reports"
- Review performance metrics (Accuracy, Precision, Recall, F1)
- Analisis confusion matrix
- Review per-class performance
- Lihat misclassifications analysis

### 6️⃣ Predict (Deployment)
- Upload gambar baru
- Klik "Classify Image"
- Lihat hasil prediksi
- Review confidence scores untuk setiap kelas

## 🎯 Tips Penggunaan

### Best Practices
1. **Ikuti Urutan:** Jalankan tahapan CRISP-DM secara berurutan untuk hasil terbaik
2. **Check Pipeline Status:** Lihat sidebar untuk status pipeline saat ini
3. **Save Progress:** Semua hasil akan disimpan otomatis di folder `models/` dan `reports/`
4. **Experiment:** Coba berbagai parameter di `config.yaml` dan train ulang model
5. **Monitor Performance:** Selalu evaluasi model setelah training

### Common Issues
- **Model Not Found:** Pastikan sudah melakukan training di tab Modeling
- **Data Not Prepared:** Jalankan Data Preparation sebelum Modeling
- **Slow Training:** GridSearchCV bisa memakan waktu, tunggu hingga selesai
- **Image Upload Failed:** Pastikan format gambar adalah JPG/JPEG/PNG

## 📁 Struktur Output

### Models Directory (`models/`)
- `model.joblib` - Trained SVM model
- `train_data.joblib` - Training features & labels
- `val_data.joblib` - Validation features & labels
- `test_data.joblib` - Test features & labels
- `*_paths.joblib` - Image file paths

### Reports Directory (`reports/`)
- `data_summary.csv` - Dataset summary
- `class_distribution.png` - Class distribution plot
- `svm_grid_results.csv` - GridSearchCV results
- `confusion_matrix.png` - Confusion matrix visualization
- `classification_report.txt` - Detailed classification report
- `misclassifications.csv` - List of misclassified images

## ⚙️ Konfigurasi

Edit `configs/config.yaml` untuk mengubah:
- **Image preprocessing:** Target size, normalization
- **Train/val/test split:** Validation size, test size
- **Model parameters:** C, kernel, gamma
- **Feature extraction:** HOG vs raw pixels
- **Grid search:** Parameter ranges untuk tuning

## 🔧 Teknologi yang Digunakan

- **Streamlit** - Web framework untuk data science apps
- **Plotly** - Interactive visualizations
- **Scikit-learn** - Machine learning (SVM, GridSearchCV)
- **Scikit-image** - Image processing (HOG features)
- **Pandas** - Data manipulation
- **Matplotlib & Seaborn** - Static visualizations
- **PIL/Pillow** - Image handling

## 📞 Support

Jika ada masalah atau pertanyaan:
1. Cek file `README.md` untuk dokumentasi dasar
2. Review konfigurasi di `configs/config.yaml`
3. Lihat logs di terminal untuk error messages
4. Pastikan semua dependencies terinstall dengan benar

## 🎉 Selamat Menggunakan!

Website ini dirancang untuk memberikan pengalaman CRISP-DM yang lengkap dan interaktif. Nikmati proses eksplorasi data, training model, dan prediksi dengan UI yang modern dan user-friendly! 🚀
