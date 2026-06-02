import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

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
    .stButton > button {
        background: transparent;
        color: #c084fc !important;
        border: 2px solid #c084fc;
        border-radius: 8px !important;
        font-family: 'SF Mono', monospace !important;
        width: 100%;
        padding: 10px;
    }
    .stButton > button:hover {
        background: rgba(192, 132, 252, 0.1);
        border-color: #a855f7;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🍜 NHẬN DIỆN MÓN ĂN VIỆT NAM</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">upload ảnh món ăn - AI nhận diện tự động</div>', unsafe_allow_html=True)

@st.cache_resource
def load_tflite_model():
    model_path = "food_model.tflite"
    if os.path.exists(model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    return None

CLASS_NAMES = [
    "banh_bao", "banh_beo", "banh_canh", "banh_chung", "banh_cuon",
    "banh_khot", "banh_mi", "banh_trang", "banh_xeo", "bun_bo_hue",
    "bun_dau_mam_tom", "bun_moc", "bun_rieu", "bun_thit_nuong", "cao_lau",
    "chao_long", "che", "com_tam", "com_tay_cam", "goi_cuon",
    "hu_tieu", "mi_quang", "pho", "xoi"
]

interpreter = load_tflite_model()

if interpreter is None:
    st.error("❌ Không tìm thấy file model! Vui lòng upload file food_model.tflite")
    
    uploaded_model = st.file_uploader("Tải lên model TFLite (file .tflite)", type=["tflite"])
    if uploaded_model is not None:
        with open("food_model.tflite", "wb") as f:
            f.write(uploaded_model.getbuffer())
        st.success("✅ Đã tải model thành công!")
        st.rerun()
else:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = input_details[0]['shape'][1]
    
    st.success(f"✅ Model đã sẵn sàng! Kích thước ảnh: {input_size}x{input_size}")
    
    st.markdown("---")
    st.markdown("### 🔮 DỰ ĐOÁN MÓN ĂN")
    
    option = st.radio("Chọn cách nhập ảnh:", ["📤 Upload ảnh", "📸 Chụp ảnh từ camera"], horizontal=True)
    
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
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(img, caption="Ảnh của bạn", use_container_width=True)
        
        if st.button("🔮 NHẬN DIỆN", use_container_width=True):
            with st.spinner("Đang phân tích ảnh..."):
                img = img.resize((input_size, input_size))
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                interpreter.set_tensor(input_details[0]['index'], img_array)
                interpreter.invoke()
                predictions = interpreter.get_tensor(output_details[0]['index'])
                
                predicted_idx = np.argmax(predictions[0])
                confidence = np.max(predictions[0])
                
                if predicted_idx < len(CLASS_NAMES):
                    food_name = CLASS_NAMES[predicted_idx].replace("_", " ").title()
                else:
                    food_name = f"Món {predicted_idx + 1}"
                
                st.markdown(f"""
                <div class="result">
                    🍽️ KẾT QUẢ: {food_name}<br>
                    📊 ĐỘ TIN CẬY: {confidence*100:.1f}%
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("📋 Xem thêm các dự đoán khác"):
                    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
                    for idx in top_3_idx:
                        name = CLASS_NAMES[idx].replace("_", " ").title() if idx < len(CLASS_NAMES) else f"Món {idx + 1}"
                        prob = predictions[0][idx] * 100
                        st.write(f"- {name}: {prob:.1f}%")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #4c1d95; font-size: 0.7rem;">
    ═══════════════════════════════════════<br>
    NHẬN DIỆN MÓN ĂN VIỆT NAM - TFLITE MODEL<br>
    ═══════════════════════════════════════
</div>
""", unsafe_allow_html=True)
