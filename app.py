import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
import zipfile
import tempfile
import time
from PIL import Image

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
st.markdown('<div class="subtitle">tải lên dataset để train model hoặc dùng model có sẵn</div>', unsafe_allow_html=True)

# ============================================
# THAM SỐ
# ============================================
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# ============================================
# HÀM TẠO MODEL (GIỮ NGUYÊN CODE CỦA BẠN)
# ============================================
def create_model(num_classes=30):
    model = Sequential([
        Input(shape=(128, 128, 3)),
        
        # Block 1
        Conv2D(32, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(32, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.25),
        
        # Block 2
        Conv2D(64, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.30),
        
        # Block 3
        Conv2D(128, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.35),
        
        # Block 4
        Conv2D(256, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.40),
        
        # Head
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ============================================
# KHỞI TẠO SESSION STATE
# ============================================
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.class_names = []
    st.session_state.model_ready = False

# ============================================
# TẢI LÊN DATASET VÀ TRAIN
# ============================================
st.markdown("---")
st.markdown("### 📁 BƯỚC 1: TẢI LÊN DATASET")

uploaded_zip = st.file_uploader("Tải lên file dataset.zip (chứa thư mục Train, Validate, Test)", type=["zip"])

def create_dataframe(directory):
    filepaths, labels = [], []
    for label in os.listdir(directory):
        class_dir = os.path.join(directory, label)
        if os.path.isdir(class_dir):
            for file in os.listdir(class_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filepaths.append(os.path.join(class_dir, file))
                    labels.append(label)
    return pd.DataFrame({'filepath': filepaths, 'label': labels})

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
        
        # Tìm đúng tên thư mục (Train hay train)
        train_folder = None
        valid_folder = None
        test_folder = None
        
        for folder in os.listdir(base_dir):
            if folder.lower() == 'train':
                train_folder = folder
            elif folder.lower() == 'validate' or folder.lower() == 'val':
                valid_folder = folder
            elif folder.lower() == 'test':
                test_folder = folder
        
        if train_folder:
            train_dir = os.path.join(base_dir, train_folder)
            train_df = create_dataframe(train_dir)
            
            valid_df = pd.DataFrame()
            if valid_folder:
                valid_dir = os.path.join(base_dir, valid_folder)
                valid_df = create_dataframe(valid_dir)
            
            test_df = pd.DataFrame()
            if test_folder:
                test_dir = os.path.join(base_dir, test_folder)
                test_df = create_dataframe(test_dir)
            
            st.success(f"✅ Đã tải dataset: Train: {len(train_df)} | Validate: {len(valid_df)} | Test: {len(test_df)}")
            
            # Lấy class names
            class_names = sorted(train_df['label'].unique())
            st.session_state.class_names = class_names
            num_classes = len(class_names)
            
            st.write(f"📋 Số loại món ăn: {num_classes}")
            st.write(f"🍽️ Các món: {', '.join(class_names[:10])}" + ("..." if len(class_names) > 10 else ""))
            
            # Tạo model
            st.session_state.model = create_model(num_classes)
            
            # Data generators
            train_datagen = ImageDataGenerator(
                rescale=1./255, rotation_range=30, width_shift_range=0.2,
                shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
            )
            valid_datagen = ImageDataGenerator(rescale=1./255)
            
            train_generator = train_datagen.flow_from_dataframe(
                train_df, x_col='filepath', y_col='label', target_size=IMAGE_SIZE,
                batch_size=BATCH_SIZE, class_mode='categorical'
            )
            
            valid_generator = None
            if len(valid_df) > 0:
                valid_generator = valid_datagen.flow_from_dataframe(
                    valid_df, x_col='filepath', y_col='label', target_size=IMAGE_SIZE,
                    batch_size=BATCH_SIZE, class_mode='categorical'
                )
            
            # ============================================
            # BƯỚC 2: TRAIN MODEL
            # ============================================
            st.markdown("---")
            st.markdown("### 🚀 BƯỚC 2: TRAIN MODEL")
            
            epochs = st.slider("Số epochs", 5, 50, 20)
            
            if st.button("BẮT ĐẦU TRAIN", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                callbacks = [
                    EarlyStopping(monitor='val_accuracy' if valid_generator else 'accuracy', 
                                 patience=5, restore_best_weights=True, verbose=0),
                    ReduceLROnPlateau(monitor='val_loss' if valid_generator else 'loss', 
                                     factor=0.5, patience=3, min_lr=1e-7, verbose=0)
                ]
                
                history = {'accuracy': [], 'loss': []}
                if valid_generator:
                    history['val_accuracy'] = []
                    history['val_loss'] = []
                
                for epoch in range(epochs):
                    status_text.text(f"Epoch {epoch+1}/{epochs} - đang train...")
                    
                    if valid_generator:
                        history_epoch = st.session_state.model.fit(
                            train_generator,
                            validation_data=valid_generator,
                            epochs=1,
                            callbacks=callbacks,
                            verbose=0
                        )
                        history['val_accuracy'].extend(history_epoch.history.get('val_accuracy', []))
                        history['val_loss'].extend(history_epoch.history.get('val_loss', []))
                    else:
                        history_epoch = st.session_state.model.fit(
                            train_generator,
                            epochs=1,
                            callbacks=callbacks,
                            verbose=0
                        )
                    
                    history['accuracy'].extend(history_epoch.history.get('accuracy', []))
                    history['loss'].extend(history_epoch.history.get('loss', []))
                    
                    progress_bar.progress((epoch + 1) / epochs)
                
                status_text.text("✅ Hoàn thành training!")
                st.session_state.model_ready = True
                
                # Vẽ biểu đồ
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                ax1.plot(history['accuracy'], label='Train')
                if valid_generator and 'val_accuracy' in history:
                    ax1.plot(history['val_accuracy'], label='Validation')
                ax1.set_title('Accuracy')
                ax1.legend()
                ax2.plot(history['loss'], label='Train')
                if valid_generator and 'val_loss' in history:
                    ax2.plot(history['val_loss'], label='Validation')
                ax2.set_title('Loss')
                ax2.legend()
                st.pyplot(fig)
                
                st.success("🎉 Model đã được train xong! Chuyển sang bước 3 để dự đoán.")
        else:
            st.error("Không tìm thấy thư mục Train trong file zip!")

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
        
        img_resized = img.resize(IMAGE_SIZE)
        img_array = img_to_array(img_resized)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        if st.button("NHẬN DIỆN", use_container_width=True):
            with st.spinner("Đang phân tích..."):
                time.sleep(0.5)
                predictions = st.session_state.model.predict(img_array)
                predicted_idx = np.argmax(predictions[0])
                confidence = np.max(predictions[0])
                
                food_name = st.session_state.class_names[predicted_idx]
                
                st.markdown(f"""
                <div class="result">
                    🍽️ KẾT QUẢ: {food_name}<br>
                    📊 ĐỘ TIN CẬY: {confidence*100:.1f}%
                </div>
                """, unsafe_allow_html=True)
