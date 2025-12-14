# CRISP-DM Image Classifier - Streamlit App

Aplikasi web untuk klasifikasi gambar menggunakan metodologi CRISP-DM dengan SVM classifier.

## 🚀 Live Demo

Upload gambar dan dapatkan prediksi real-time untuk kategori:
- **Black**
- **Brown** 
- **White**

## 📋 Fitur

- ✨ Upload gambar untuk prediksi instant
- 🖼️ Galeri contoh dari validation set
- 📊 Visualisasi probabilitas prediksi interaktif
- 📈 Performance metrics dan confusion matrix
- 🎨 UI modern dengan tema terang

## 🛠️ Teknologi

- **Python 3.10+**
- **Streamlit** - Web framework
- **Scikit-learn** - Machine Learning (SVM)
- **Scikit-image** - Image processing (HOG features)
- **Plotly** - Interactive charts

## 📦 Instalasi Lokal

1. Clone repository:
```bash
git clone <your-repo-url>
cd "data saints"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Jalankan aplikasi:
```bash
streamlit run streamlit_app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

## 🌐 Deploy ke Streamlit Community Cloud

### Langkah 1: Setup GitHub Repository

```bash
git init
git add .
git commit -m "Initial commit - CRISP-DM Image Classifier"
git branch -M main
git remote add origin https://github.com/username/repo-name.git
git push -u origin main
```

### Langkah 2: Deploy di Streamlit Cloud

1. Login ke [share.streamlit.io](https://share.streamlit.io/)
2. Klik **"New app"**
3. Pilih:
   - **Repository**: Your GitHub repository
   - **Branch**: main
   - **Main file path**: `streamlit_app.py`
4. Klik **"Deploy!"**

### ✅ File yang Diperlukan:

- ✅ `streamlit_app.py` - Main application
- ✅ `requirements.txt` - Python dependencies
- ✅ `models/*.joblib` - Trained model files
- ✅ `configs/config.yaml` - Configuration
- ✅ `.streamlit/config.toml` - Streamlit config
- ✅ `train/` folders - Training data
- ✅ `reports/` - Performance reports

## 📁 Struktur Project

```
.
├── streamlit_app.py              # 🌐 Main web application
├── requirements.txt              # 📦 Python dependencies
├── .streamlit/
│   └── config.toml              # ⚙️ Streamlit configuration
├── configs/
│   └── config.yaml              # 🔧 Model configuration
├── models/                       # 🤖 Model files
│   ├── model.joblib             # Trained SVM model
│   ├── val_paths.joblib         # Validation image paths
│   └── val_data.joblib          # Validation features
├── reports/                      # 📊 Performance reports
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── svm_grid_results.csv
├── src/                          # 💻 Source modules
│   ├── main.py                  # Full training pipeline
│   ├── predict.py               # CLI prediction tool
│   ├── data_understanding.py
│   ├── data_preparation.py
│   ├── modeling.py
│   └── evaluation.py
└── train/                        # 🖼️ Training dataset
    ├── Black/
    ├── Brown/
    └── White/
```

## 📊 Model Performance

- **Validation Accuracy**: 76.0%
- **Algorithm**: Support Vector Machine (SVM)
- **Features**: HOG (Histogram of Oriented Gradients)
- **Kernel**: RBF with GridSearchCV optimization
- **Image Size**: 128x128 pixels

## 🎯 CRISP-DM Methodology

1. **Business Understanding**: Classify images into 3 color categories
2. **Data Understanding**: Dataset analysis, class distribution
3. **Data Preparation**: Preprocessing, HOG feature extraction
4. **Modeling**: SVM training with hyperparameter tuning
5. **Evaluation**: Confusion matrix, classification metrics
6. **Deployment**: ✅ Interactive web application

## 🔧 Training Pipeline (Optional)

Untuk melatih model dari awal:

```bash
# Run full CRISP-DM pipeline
python -m src.main

# Predict menggunakan CLI
python -m src.predict "path/to/image.jpg"
```

## ⚙️ Configuration

Edit `configs/config.yaml` untuk mengubah:
- Target image size
- Feature extraction method (HOG/raw)
- SVM hyperparameters
- Train/validation/test split ratios

## 🐛 Troubleshooting

### Error: ModuleNotFoundError
```bash
pip install -r requirements.txt
```

### Error: Model tidak ditemukan
Pastikan folder `models/` berisi file `model.joblib`

### Error: Permission denied
Jalankan dengan administrator atau ubah path direktori

## 📝 License

MIT License

---

**Made with ❤️ using Streamlit and CRISP-DM methodology**
