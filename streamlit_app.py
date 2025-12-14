import streamlit as st
from pathlib import Path
import yaml
import pandas as pd
from joblib import load
from PIL import Image
import numpy as np
from skimage import io, color, transform, feature
import plotly.graph_objects as go
import random

# Simple page config
st.set_page_config(
    page_title="CRISP-DM Image Classifier", 
    page_icon="🎯", 
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Force light theme
st.markdown("""
<script>
    window.parent.document.body.classList.add('light-theme');
</script>
""", unsafe_allow_html=True)

# Light theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* Force light background */
    .stApp {
        background-color: #ffffff !important;
    }
    
    [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
    }
    
    [data-testid="stHeader"] {
        background-color: #ffffff !important;
    }
    
    /* Font and text colors */
    * { 
        font-family: 'Inter', sans-serif !important;
        color: #1f2937;
    }
    
    /* Title and headings */
    .main-title { 
        font-size: 2rem; 
        font-weight: 700; 
        color: #1f2937 !important; 
        margin-bottom: 0.5rem; 
    }
    
    .section-title { 
        font-size: 1.25rem; 
        font-weight: 600; 
        color: #374151 !important; 
        margin: 1.5rem 0 1rem 0; 
    }
    
    /* Status badge */
    .status-badge { 
        display: inline-block; 
        background: #10b981; 
        color: white !important; 
        padding: 0.25rem 0.75rem; 
        border-radius: 1rem; 
        font-size: 0.875rem; 
        font-weight: 600; 
        float: right; 
    }
    
    /* Buttons */
    .stButton>button { 
        border-radius: 0.5rem; 
        font-weight: 600; 
        padding: 0.5rem 2rem;
        background-color: #6366f1;
        color: white !important;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #4f46e5;
        border: none;
    }
    
    /* Prediction box */
    .prediction-box { 
        background: #f9fafb !important; 
        padding: 2rem; 
        border-radius: 0.75rem; 
        text-align: center; 
        margin: 1rem 0;
        border: 1px solid #e5e7eb;
    }
    
    .prediction-result { 
        font-size: 2.5rem; 
        font-weight: 700; 
        color: #7c3aed !important; 
        margin: 1rem 0; 
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] { 
        border: 2px dashed #d1d5db; 
        border-radius: 0.5rem; 
        padding: 1.5rem;
        background-color: #ffffff !important;
    }
    
    /* Divider */
    hr {
        border-color: #e5e7eb !important;
    }
    
    /* Info boxes */
    [data-testid="stMarkdownContainer"] p {
        color: #374151 !important;
    }
    
    /* Ensure plotly charts have light background */
    .js-plotly-plot {
        background-color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Helpers
ROOT = Path(__file__).resolve().parent
CFG_PATH = ROOT / "configs" / "config.yaml"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

def load_config():
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def preprocess_for_predict(image_path: str, cfg: dict):
    img = io.imread(image_path)
    if img.ndim == 2:
        img = color.gray2rgb(img)
    target_size = tuple(cfg["images"]["target_size"])
    img = transform.resize(img, target_size, anti_aliasing=True)
    if cfg["images"]["normalize"]:
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    if cfg["model"]["features"] == "raw":
        vec = img.reshape(-1)
    else:
        gray = color.rgb2gray(img)
        vec = feature.hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9, block_norm="L2-Hys")
    return vec

def get_sample_images(n=12):
    val_paths_file = MODELS_DIR / "val_paths.joblib"
    val_data_file = MODELS_DIR / "val_data.joblib"
    if val_paths_file.exists() and val_data_file.exists():
        val_paths = load(val_paths_file)
        _, y_val = load(val_data_file)
        samples = []
        for label in ["Black", "Brown", "White"]:
            label_indices = [i for i, l in enumerate(y_val) if l == label]
            if label_indices:
                selected = random.sample(label_indices, min(4, len(label_indices)))
                for idx in selected:
                    samples.append((val_paths[idx], label))
        random.shuffle(samples)
        return samples[:n]
    return []

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<div class="main-title">CRISP-DM Image Classifier</div>', unsafe_allow_html=True)
with col2:
    model_file = MODELS_DIR / "model.joblib"
    if model_file.exists():
        st.markdown('<span class="status-badge">● Model Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge" style="background: #ef4444;">● No Model</span>', unsafe_allow_html=True)

st.markdown("---")

cfg = load_config()

# Main Content
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown('<div class="section-title">Unggah Gambar atau Pilih Contoh</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose File", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        predict_button = st.button("Prediksi", width="stretch", type="primary")
    with col_btn2:
        clear_button = st.button("Bersihkan", width="stretch")
    
    if uploaded_file is not None and not clear_button:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, use_column_width=True)
        
        if predict_button:
            model_file = MODELS_DIR / "model.joblib"
            if model_file.exists():
                clf = load(model_file)
                tmp_path = ROOT / "_tmp_upload.jpg"
                img.save(tmp_path)
                vec = preprocess_for_predict(str(tmp_path), cfg)
                pred = clf.predict([vec])[0]
                proba = None
                if hasattr(clf, 'predict_proba'):
                    try:
                        proba = clf.predict_proba([vec])[0]
                        classes = clf.classes_
                    except:
                        proba = None
                tmp_path.unlink(missing_ok=True)
                st.session_state['prediction'] = pred
                st.session_state['probabilities'] = proba
                st.session_state['classes'] = classes if proba is not None else None
            else:
                st.error("Model tidak ditemukan!")
    
    st.markdown('<div class="section-title">Contoh dari Validation Set</div>', unsafe_allow_html=True)
    sample_images = get_sample_images(12)
    if sample_images:
        cols = st.columns(6)
        for idx, (img_path, label) in enumerate(sample_images):
            with cols[idx % 6]:
                if Path(img_path).exists():
                    try:
                        img = Image.open(img_path)
                        st.image(img, use_column_width=True)
                        st.markdown(f'<div style="text-align: center; font-size: 0.75rem; color: #6b7280;">{label}</div>', unsafe_allow_html=True)
                    except:
                        pass
    else:
        st.info("Tidak ada data validasi.")

with col_right:
    st.markdown('<div class="section-title">Hasil Prediksi</div>', unsafe_allow_html=True)
    
    if 'prediction' in st.session_state and st.session_state['prediction']:
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.markdown(f'<div class="prediction-result">{st.session_state["prediction"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.session_state.get('probabilities') is not None and st.session_state.get('classes') is not None:
            proba = st.session_state['probabilities']
            classes = st.session_state['classes']
            fig = go.Figure(data=[go.Bar(
                x=classes, 
                y=proba, 
                marker=dict(color='#a78bfa'), 
                text=[f'{p:.2%}' for p in proba], 
                textposition='auto',
                textfont=dict(size=14, color='#1f2937', family='Inter'),
                name='Probabilitas'
            )])
            fig.update_layout(
                template='plotly_white',
                paper_bgcolor='white',
                plot_bgcolor='white',
                xaxis_title="", 
                yaxis_title="", 
                xaxis=dict(tickfont=dict(size=12, color='#1f2937', family='Inter')),
                yaxis=dict(range=[0, 1], tickformat='.0%', tickfont=dict(size=12, color='#1f2937', family='Inter')), 
                height=300, 
                showlegend=True, 
                legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="right", x=1, font=dict(size=12, color='#1f2937', family='Inter')),
                font=dict(size=12, color='#1f2937', family='Inter'),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, width="stretch")
    else:
        st.info("Belum ada prediksi.")
        fig = go.Figure(data=[go.Bar(
            x=['Black', 'Brown', 'White'], 
            y=[0, 0, 0], 
            marker=dict(color='#a78bfa'),
            textfont=dict(size=14, color='#1f2937', family='Inter'),
            name='Probabilitas'
        )])
        fig.update_layout(
            template='plotly_white',
            paper_bgcolor='white',
            plot_bgcolor='white',
            xaxis_title="", 
            yaxis_title="", 
            xaxis=dict(tickfont=dict(size=12, color='#1f2937', family='Inter')),
            yaxis=dict(range=[0, 1], tickformat='.0%', tickfont=dict(size=12, color='#1f2937', family='Inter')), 
            height=300, 
            showlegend=True, 
            legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="right", x=1, font=dict(size=12, color='#1f2937', family='Inter')),
            font=dict(size=12, color='#1f2937', family='Inter'),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, width="stretch")

if clear_button:
    for key in ['prediction', 'probabilities', 'classes']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

st.markdown("---")

# Bottom Section
col_perf, col_cm = st.columns([1, 1])

with col_perf:
    st.markdown('<div class="section-title">Performa Model</div>', unsafe_allow_html=True)
    cr_path = REPORTS_DIR / "classification_report.txt"
    if cr_path.exists():
        import re
        report_text = Path(cr_path).read_text(encoding="utf-8")
        accuracy_match = re.search(r'accuracy\s+(\d+\.\d+)', report_text)
        if accuracy_match:
            accuracy = float(accuracy_match.group(1))
            st.markdown(f'<div style="font-size: 1rem; margin-bottom: 1rem;">Best Validation Accuracy: <strong>{accuracy:.1%}</strong></div>', unsafe_allow_html=True)
    else:
        st.info("Belum ada hasil evaluasi.")
    
    grid_path = REPORTS_DIR / "svm_grid_results.csv"
    if grid_path.exists():
        df_grid = pd.read_csv(grid_path)
        top_results = df_grid.sort_values('rank_test_score').head(10)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(1, len(top_results) + 1)), 
            y=top_results['mean_test_score'], 
            marker=dict(color=['#10b981' if i == 0 else '#6366f1' for i in range(len(top_results))]), 
            text=top_results['mean_test_score'].apply(lambda x: f'{x:.3f}'), 
            textposition='auto',
            textfont=dict(size=12, color='#1f2937', family='Inter'),
            name='Val Acc'
        ))
        fig.add_trace(go.Scatter(
            x=list(range(1, len(top_results) + 1)), 
            y=[0.85] * len(top_results), 
            mode='lines+markers', 
            name='Train Acc', 
            line=dict(color='#3b82f6', width=2)
        ))
        fig.update_layout(
            template='plotly_white',
            paper_bgcolor='white',
            plot_bgcolor='white',
            xaxis_title="Configuration Rank",
            yaxis_title="Accuracy",
            xaxis=dict(
                title=dict(font=dict(size=12, color='#1f2937', family='Inter')),
                tickfont=dict(size=11, color='#1f2937', family='Inter')
            ),
            yaxis=dict(
                title=dict(font=dict(size=12, color='#1f2937', family='Inter')),
                tickfont=dict(size=11, color='#1f2937', family='Inter')
            ),
            height=250, 
            showlegend=True, 
            legend=dict(orientation="h", yanchor="top", y=1.15, xanchor="right", x=1, font=dict(size=11, color='#1f2937', family='Inter')),
            font=dict(size=11, color='#1f2937', family='Inter'),
            margin=dict(l=20, r=20, t=20, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

with col_cm:
    st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
    cm_path = REPORTS_DIR / "confusion_matrix.png"
    if cm_path.exists():
        st.image(str(cm_path), use_column_width=True)
    else:
        st.info("Belum ada confusion matrix.")
