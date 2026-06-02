import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os

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
st.markdown('<div class="subtitle">upload ảnh món ăn - AI nhận diện tự động</div>', unsafe_allow_html=True)

# ============================================
# TẢI MODEL .H5 (TENSORFLOW)
# ============================================
@st.cache_resource
def load_cnn_model():
    """Load model CNN từ file .h5"""
    model_path = "food_model.h5"
    if os.path.exists(model_path):
        model = load_model(model_path)
        return model
    return None

# ============================================
# DANH SÁCH MÓN ĂN (CẬP NHẬT THEO MODEL CỦA BẠN)
# ============================================
CLASS_NAMES = [
    "banh_bao", "banh_beo", "banh_canh", "banh_chung", "banh_cuon",
    "banh_khot", "banh_mi", "banh_trang", "banh_xeo", "bun_bo_hue",
    "bun_dau_mam_tom", "bun_moc", "bun_rieu", "bun_thit_nuong", "cao_lau",
    "chao_long", "che", "com_tam", "com_tay_cam", "goi_cuon",
    "hu_tieu", "mi_quang", "pho", "xoi"
]

model = load_cnn_model()

if model is None:
    st.info("💡 Chưa có model. Vui lòng upload file food_model.h5!")
    
    uploaded_model = st.file_uploader("Tải lên model CNN (file .h5)", type=["h5"])
    if uploaded_model is not None:
        with open("food_model.h5", "wb") as f:
            f.write(uploaded_model.getbuffer())
        st.success("✅ Đã tải model CNN thành công!")
        st.rerun()
else:
    st.success("✅ Model CNN đã sẵn sàng!")
    
    # ============================================
    # DỰ ĐOÁN
    # ============================================
    st.markdown("---")
    st.markdown("### 🔮 DỰ ĐOÁN MÓN ĂN")
    
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
            with st.spinner("Đang phân tích với CNN..."):
                # Tiền xử lý ảnh (cùng kích thước với lúc train)
                img = img.resize((128, 128))
                img_array = img_to_array(img)
                img_array = img_array / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Dự đoán
                predictions = model.predict(img_array)
                predicted_idx = np.argmax(predictions[0])
                confidence = np.max(predictions[0])
                
                food_name = CLASS_NAMES[predicted_idx].replace("_", " ").title()
                
                st.markdown(f"""
                <div class="result">
                    🍽️ KẾT QUẢ: {food_name}<br>
                    📊 ĐỘ TIN CẬY: {confidence*100:.1f}%
                </div>
                """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #4c1d95; font-size: 0.7rem;">
    ═══════════════════════════════════════<br>
    NHẬN DIỆN MÓN ĂN VIỆT NAM - CNN MODEL<br>
    ═══════════════════════════════════════
</div>
""", unsafe_allow_html=True)
