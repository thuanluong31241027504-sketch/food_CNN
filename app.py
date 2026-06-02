import streamlit as st
import numpy as np
from PIL import Image
import pickle
import gzip

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
# HÀM TRÍCH XUẤT ĐẶC TRƯNG (THAY CHO TENSORFLOW)
# ============================================
def extract_features(img_path, img_size=(64, 64)):
    """Trích xuất đặc trưng từ ảnh"""
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(img_size)
        img_array = np.array(img)
        
        # Đặc trưng màu sắc
        hist_r = np.histogram(img_array[:,:,0], bins=16, range=(0, 256))[0]
        hist_g = np.histogram(img_array[:,:,1], bins=16, range=(0, 256))[0]
        hist_b = np.histogram(img_array[:,:,2], bins=16, range=(0, 256))[0]
        
        # Grayscale
        gray = np.dot(img_array[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
        hist_gray = np.histogram(gray, bins=16, range=(0, 256))[0]
        
        # Edge features
        gx = np.gradient(gray, axis=1)
        gy = np.gradient(gray, axis=0)
        edge_magnitude = np.sqrt(gx**2 + gy**2)
        hist_edge = np.histogram(edge_magnitude.flatten(), bins=16, range=(0, 50))[0]
        
        features = np.concatenate([hist_r, hist_g, hist_b, hist_gray, hist_edge])
        features = features / (features.sum() + 1e-7)
        
        return features
    except Exception as e:
        return None

# ============================================
# TẢI MODEL (DÙNG PICKLE)
# ============================================
@st.cache_resource
def load_model():
    """Load model đã train từ file pickle"""
    import os
    model_path = "food_model.pkl"
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
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

model = load_model()

if model is None:
    st.info("💡 Chưa có model. Vui lòng upload model file!")
    
    uploaded_model = st.file_uploader("Tải lên model (file .pkl)", type=["pkl"])
    if uploaded_model is not None:
        with open("food_model.pkl", "wb") as f:
            f.write(uploaded_model.getbuffer())
        st.success("✅ Đã tải model!")
        st.rerun()
else:
    st.success("✅ Model đã sẵn sàng!")
    
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
            with st.spinner("Đang phân tích..."):
                # Lưu ảnh tạm
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                    img.save(tmp_file.name)
                    features = extract_features(tmp_file.name)
                    import os
                    os.unlink(tmp_file.name)
                
                if features is not None:
                    features = features.reshape(1, -1)
                    
                    # Dự đoán
                    prediction = model.predict(features)[0]
                    if hasattr(model, 'predict_proba'):
                        confidence_probs = model.predict_proba(features)[0]
                        confidence = np.max(confidence_probs)
                    else:
                        confidence = 0.8
                    
                    food_name = CLASS_NAMES[prediction].replace("_", " ").title()
                    
                    st.markdown(f"""
                    <div class="result">
                        🍽️ KẾT QUẢ: {food_name}<br>
                        📊 ĐỘ TIN CẬY: {confidence*100:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("Không thể phân tích ảnh!")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #4c1d95; font-size: 0.7rem;">
    ═══════════════════════════════════════<br>
    NHẬN DIỆN MÓN ĂN VIỆT NAM - AI SYSTEM<br>
    ═══════════════════════════════════════
</div>
""", unsafe_allow_html=True)
