import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import zipfile
import tempfile
import time
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# ============================================
# CẤU HÌNH TRANG
# ============================================
st.set_page_config(
    page_title="NHẬN DIỆN MÓN ĂN VIỆT NAM",
    page_icon="🍜",
    layout="centered"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a0a2e 50%, #0a0a1a 100%);
        min-height: 100vh;
    }
    @import url('https://fonts.googleapis.com/css2?family=SF+Mono&display=swap');
    .title {
        font-family: 'SF Mono', monospace;
        font-size: 2rem;
        text-align: center;
        color: #c084fc;
        margin-top: 40px;
    }
    .subtitle {
        font-family: 'SF Mono', monospace;
        font-size: 0.8rem;
        text-align: center;
        color: #8b5cf6;
        margin-bottom: 40px;
    }
    .stButton > button {
        background: transparent;
        color: #c084fc !important;
        border: 2px solid #c084fc;
        border-radius: 8px !important;
        font-family: 'SF Mono', monospace !important;
    }
    .result {
        font-family: 'SF Mono', monospace;
        font-size: 1.2rem;
        text-align: center;
        color: #c084fc;
        padding: 20px;
        border: 1px solid #c084fc;
        border-radius: 8px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🍜 NHẬN DIỆN MÓN ĂN VIỆT NAM</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">tải lên dataset để train model - nhận diện món ăn từ ảnh</div>', unsafe_allow_html=True)

# ============================================
# KHỞI TẠO SESSION STATE
# ============================================
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.label_encoder = None
    st.session_state.class_names = []
    st.session_state.model_ready = False

# ============================================
# HÀM TRÍCH XUẤT ĐẶC TRƯNG TỪ ẢNH
# ============================================
def extract_features(img_path, img_size=(64, 64)):
    """Trích xuất đặc trưng cơ bản từ ảnh"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, img_size)
    
    # Chuyển sang HSV để trích xuất màu sắc
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Đặc trưng màu sắc (histogram)
    hist_hue = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    hist_sat = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    hist_val = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    
    # Đặc trưng kết cấu (gray)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist_gray = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
    
    # Đặc trưng cạnh (Sobel)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    hist_edge = cv2.calcHist([edge_magnitude.astype(np.uint8)], [0], None, [16], [0, 256]).flatten()
    
    # Kết hợp tất cả đặc trưng
    features = np.concatenate([hist_hue, hist_sat, hist_val, hist_gray, hist_edge])
    
    return features / (features.sum() + 1e-7)

# ============================================
# HÀM LOAD DATASET
# ============================================
def load_dataset_from_folder(folder_path):
    """Load ảnh từ folder và trích xuất đặc trưng"""
    filepaths = []
    labels = []
    
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filepaths.append(os.path.join(label_path, file))
                    labels.append(label)
    
    if len(filepaths) == 0:
        return None, None
    
    # Trích xuất đặc trưng cho từng ảnh
    features = []
    with st.spinner("Đang trích xuất đặc trưng từ ảnh..."):
        progress_bar = st.progress(0)
        for i, fp in enumerate(filepaths):
            feat = extract_features(fp)
            if feat is not None:
                features.append(feat)
            progress_bar.progress((i + 1) / len(filepaths))
    
    return np.array(features), np.array(labels)

# ============================================
# BƯỚC 1: TẢI LÊN DATASET
# ============================================
st.markdown("---")
st.markdown("### 📁 BƯỚC 1: TẢI LÊN DATASET")

uploaded_zip = st.file_uploader("Tải lên file dataset.zip (chứa thư mục Train, Validate, Test)", type=["zip"])

if uploaded_zip is not None:
    with tempfile.TemporaryDirectory() as tmpdir:
        with st.spinner("Đang giải nén dataset..."):
            zip_path = os.path.join(tmpdir, "dataset.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.getbuffer())
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            
            # Tìm thư mục gốc
            base_dir = tmpdir
            for item in os.listdir(tmpdir):
                item_path = os.path.join(tmpdir, item)
                if os.path.isdir(item_path):
                    if 'Train' in os.listdir(item_path) or 'train' in [x.lower() for x in os.listdir(item_path)]:
                        base_dir = item_path
                        break
        
        # Tìm các thư mục
        train_folder = None
        for folder in os.listdir(base_dir):
            if folder.lower() == 'train':
                train_folder = folder
                break
        
        if train_folder:
            train_dir = os.path.join(base_dir, train_folder)
            
            st.info("Đang load dữ liệu train...")
            X_train, y_train = load_dataset_from_folder(train_dir)
            
            if X_train is not None:
                st.success(f"✅ Đã load {len(X_train)} ảnh train với {len(np.unique(y_train))} loại món")
                
                st.session_state.class_names = sorted(np.unique(y_train))
                
                st.write(f"📋 Các món ăn: {', '.join(st.session_state.class_names)}")
                
                # ============================================
                # BƯỚC 2: TRAIN MODEL
                # ============================================
                st.markdown("---")
                st.markdown("### 🚀 BƯỚC 2: TRAIN MODEL")
                
                # Encode labels
                label_encoder = LabelEncoder()
                y_train_encoded = label_encoder.fit_transform(y_train)
                st.session_state.label_encoder = label_encoder
                
                # Chia train/val
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_train, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
                )
                
                st.write(f"📊 Train: {len(X_train_split)} ảnh | Validation: {len(X_val_split)} ảnh")
                
                # Tham số model
                n_estimators = st.slider("Số cây (n_estimators)", 50, 300, 100, step=50)
                max_depth = st.slider("Độ sâu tối đa (max_depth)", 10, 50, 20, step=5)
                
                if st.button("🚀 BẮT ĐẦU TRAIN", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Train Random Forest
                    status_text.text("Đang train Random Forest...")
                    
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=42,
                        n_jobs=-1
                    )
                    
                    model.fit(X_train_split, y_train_split)
                    
                    progress_bar.progress(0.8)
                    status_text.text("Đang đánh giá model...")
                    
                    # Đánh giá
                    y_pred = model.predict(X_val_split)
                    accuracy = accuracy_score(y_val_split, y_pred)
                    
                    st.session_state.model = model
                    st.session_state.model_ready = True
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Hoàn thành!")
                    
                    st.success(f"🎉 Accuracy trên validation: {accuracy*100:.2f}%")
                    
                    # Hiển thị báo cáo
                    st.markdown("#### 📊 Báo cáo chi tiết:")
                    report = classification_report(y_val_split, y_pred, target_names=st.session_state.class_names, output_dict=True)
                    
                    # Chuyển thành DataFrame để hiển thị
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df)
                    
                    # Lưu model
                    model_path = os.path.join(tmpdir, "model.pkl")
                    with open(model_path, "wb") as f:
                        pickle.dump((model, label_encoder, st.session_state.class_names), f)
                    
                    st.info("💡 Model đã sẵn sàng! Chuyển sang Bước 3 để dự đoán.")
            else:
                st.error("Không thể load dữ liệu từ folder Train!")
        else:
            st.error("Không tìm thấy thư mục Train!")

# ============================================
# BƯỚC 3: DỰ ĐOÁN
# ============================================
if st.session_state.model_ready and st.session_state.model is not None:
    st.markdown("---")
    st.markdown("### 🔮 BƯỚC 3: DỰ ĐOÁN MÓN ĂN")
    
    option = st.radio("Chọn cách nhập ảnh:", ["📤 Upload ảnh", "📸 Chụp ảnh từ camera"])
    
    img = None
    
    if option == "📤 Upload ảnh":
        uploaded_file = st.file_uploader("Chọn ảnh món ăn", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
    else:
        camera_img = st.camera_input("Chụp ảnh món ăn")
        if camera_img is not None:
            img = Image.open(camera_img)
    
    if img is not None:
        st.image(img, caption="Ảnh của bạn", use_container_width=True)
        
        if st.button("🔮 NHẬN DIỆN", use_container_width=True):
            with st.spinner("Đang phân tích..."):
                # Lưu ảnh tạm
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                    img.save(tmp_file.name)
                    # Trích xuất đặc trưng
                    features = extract_features(tmp_file.name)
                    os.unlink(tmp_file.name)
                
                if features is not None:
                    features = features.reshape(1, -1)
                    prediction = st.session_state.model.predict(features)[0]
                    confidence_probs = st.session_state.model.predict_proba(features)[0]
                    confidence = np.max(confidence_probs)
                    
                    food_name = st.session_state.class_names[prediction]
                    
                    st.markdown(f"""
                    <div class="result">
                        🍽️ KẾT QUẢ: {food_name}<br>
                        📊 ĐỘ TIN CẬY: {confidence*100:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("Không thể phân tích ảnh. Vui lòng thử ảnh khác!")
else:
    if uploaded_zip is None:
        st.info("💡 Hãy tải lên dataset và train model trước khi dự đoán!")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #4c1d95; font-size: 0.7rem;">
    ═══════════════════════════════════════<br>
    NHẬN DIỆN MÓN ĂN VIỆT NAM - AI SYSTEM<br>
    ═══════════════════════════════════════
</div>
""", unsafe_allow_html=True)
