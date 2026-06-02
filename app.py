import streamlit as st
import os
import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Dense, Flatten, Dropout, BatchNormalization,
    GlobalAveragePooling2D, Input, Conv2D, MaxPooling2D
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import kagglehub
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Vietnamese Food Recognition 🍜",
    page_icon="🍜",
    layout="wide"
)

# Constants (GIỮ NGUYÊN từ code Colab)
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 30
IMAGE_SIZE = (128, 128)
LR_TRANSFER = LEARNING_RATE
BASE_DIRS = []

# Hàm tạo dataframe và gộp dataset (GIỮ NGUYÊN)
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

def merge_datasets(base_dirs, subset):
    dfs = []
    for base_dir in base_dirs:
        subset_dir = os.path.join(base_dir, subset)
        if os.path.exists(subset_dir):
            dfs.append(create_dataframe(subset_dir))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=['filepath', 'label'])

@st.cache_resource
def load_and_train():
    """Load dữ liệu và train model - chỉ chạy 1 lần"""
    
    with st.spinner("📥 Đang tải dữ liệu từ Kaggle..."):
        # Download dataset từ kagglehub
        path = kagglehub.dataset_download("quandang/vietnamese-foods")
        BASE_DIRS.append(path)
        
        # Load dữ liệu (GIỮ NGUYÊN cấu trúc)
        train_df = merge_datasets(BASE_DIRS, 'Train')
        valid_df = merge_datasets(BASE_DIRS, 'Validate')
        test_df = merge_datasets(BASE_DIRS, 'Test')
        
        # Nếu không có validation, split từ train
        if len(valid_df) == 0 and len(train_df) > 0:
            train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=42)
    
    st.success(f"✅ Train: {len(train_df)} | Validate: {len(valid_df)} | Test: {len(test_df)}")
    
    # Data Augmentation (GIỮ NGUYÊN)
    train_datagen = ImageDataGenerator(
        rescale=1./255, 
        rotation_range=30, 
        width_shift_range=0.2, 
        shear_range=0.2, 
        zoom_range=0.2, 
        horizontal_flip=True, 
        fill_mode='nearest'
    )
    valid_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Tạo generators
    train_generator = train_datagen.flow_from_dataframe(
        train_df, 
        x_col='filepath', 
        y_col='label', 
        target_size=IMAGE_SIZE, 
        batch_size=BATCH_SIZE, 
        class_mode='categorical'
    )
    valid_generator = valid_datagen.flow_from_dataframe(
        valid_df, 
        x_col='filepath', 
        y_col='label', 
        target_size=IMAGE_SIZE, 
        batch_size=BATCH_SIZE, 
        class_mode='categorical'
    )
    test_generator = test_datagen.flow_from_dataframe(
        test_df, 
        x_col='filepath', 
        y_col='label', 
        target_size=IMAGE_SIZE, 
        batch_size=BATCH_SIZE, 
        class_mode='categorical'
    )
    
    # Lấy danh sách classes
    CLASS_NAMES = sorted(train_generator.class_indices.keys())
    
    # Xây dựng model (GIỮ NGUYÊN cấu trúc từ Colab)
    with st.spinner("🏗️ Đang xây dựng model CNN..."):
        model = Sequential([
            Input(shape=(224, 224, 3)),

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

            # Block 4 (thêm chiều sâu)
            Conv2D(256, (3,3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(),
            Dropout(0.40),

            # Head
            GlobalAveragePooling2D(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(30, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE), 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
    
    # Callbacks (GIỮ NGUYÊN)
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=6,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Training
    with st.spinner(f"🎓 Đang training model với {EPOCHS} epochs (mất 10-15 phút)..."):
        history = model.fit(
            train_generator, 
            epochs=EPOCHS, 
            validation_data=valid_generator, 
            callbacks=callbacks,
            verbose=0
        )
    
    st.success("✅ Training hoàn tất!")
    
    return model, CLASS_NAMES, history, test_generator

# Hàm dự đoán (giữ nguyên phong cách từ Colab)
def predict_image(image, model, class_names):
    """Dự đoán loại ảnh"""
    from PIL import Image as PILImage
    
    # Resize về 224x224 như model yêu cầu
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)
    
    prediction = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    
    return class_names[predicted_class], confidence, prediction[0]

# Main UI
st.title("🍜 Vietnamese Food Recognition")
st.markdown("### Nhận diện món ăn Việt Nam bằng Deep Learning CNN")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1999/1999625.png", width=100)
    st.markdown("## 📌 Thông tin")
    st.markdown(f"""
    **Model Architecture:**
    - 4 Convolutional Blocks
    - BatchNormalization
    - Dropout regularization
    - GlobalAveragePooling2D
    - Dense layers
    
    **Parameters:**
    - Batch Size: {BATCH_SIZE}
    - Learning Rate: {LEARNING_RATE}
    - Epochs: {EPOCHS}
    - Input size: 224x224
    
    **Status:** 
    - Lần đầu chạy sẽ train model
    - Các lần sau dùng cache
    """)
    
    if st.button("🔄 Train lại model", type="primary"):
        st.cache_resource.clear()
        st.rerun()

# Khởi tạo session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Load và train model
if not st.session_state.model_loaded:
    with st.status("🚀 Đang khởi tạo (lần đầu mất 10-15 phút)...", expanded=True) as status:
        model, class_names, history, test_generator = load_and_train()
        st.session_state.model = model
        st.session_state.class_names = class_names
        st.session_state.history = history
        st.session_state.test_generator = test_generator
        st.session_state.model_loaded = True
        status.update(label="✅ Model đã sẵn sàng!", state="complete")

# Hiển thị thông tin training
if st.session_state.model_loaded:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Final Accuracy", f"{st.session_state.history.history['val_accuracy'][-1]:.2%}")
    with col2:
        st.metric("📈 Best Accuracy", f"{max(st.session_state.history.history['val_accuracy']):.2%}")
    with col3:
        st.metric("🔄 Epochs trained", len(st.session_state.history.history['val_accuracy']))
    
    # Vẽ biểu đồ training
    with st.expander("📊 Xem biểu đồ training", expanded=False):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(st.session_state.history.history['accuracy'], label='Train', linewidth=2)
        ax1.plot(st.session_state.history.history['val_accuracy'], label='Validation', linewidth=2)
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(st.session_state.history.history['loss'], label='Train', linewidth=2)
        ax2.plot(st.session_state.history.history['val_loss'], label='Validation', linewidth=2)
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    # Upload ảnh
    st.markdown("## 📤 Upload ảnh món ăn")
    
    uploaded_file = st.file_uploader(
        "Chọn file ảnh...",
        type=['jpg', 'jpeg', 'png', 'webp'],
        help="Upload ảnh món ăn Việt Nam"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = plt.imread(uploaded_file)
            st.image(image, caption="Ảnh đã upload", use_column_width=True)
        
        with col2:
            with st.spinner("🔍 Đang phân tích..."):
                from PIL import Image as PILImage
                pil_image = PILImage.open(uploaded_file)
                food_name, confidence, all_probs = predict_image(
                    pil_image, 
                    st.session_state.model, 
                    st.session_state.class_names
                )
            
            st.success("✅ Kết quả dự đoán:")
            
            # Hiển thị kết quả chính
            st.markdown(f"""
            <div style="
                padding: 20px;
                border-radius: 15px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                text-align: center;
                margin: 20px 0;
            ">
                <h1 style="margin: 0;">🍽️ {food_name}</h1>
                <p style="font-size: 20px; margin: 10px 0 0 0;">Độ tin cậy: {confidence:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Hiển thị top 5 dự đoán
            st.markdown("### 📊 Top 5 dự đoán:")
            top_5_idx = np.argsort(all_probs)[-5:][::-1]
            
            for idx in top_5_idx:
                prob = all_probs[idx]
                st.markdown(f"""
                <div style="margin: 10px 0;">
                    <b>{st.session_state.class_names[idx]}</b>
                </div>
                """, unsafe_allow_html=True)
                st.progress(prob, text=f"Confidence: {prob:.1%}")
    
    # Hiển thị danh sách món ăn
    with st.expander("🍽️ Danh sách 30 món ăn có thể nhận diện"):
        cols = st.columns(5)
        for idx, food in enumerate(st.session_state.class_names):
            cols[idx % 5].markdown(f"• {food}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray;">
        <p>🎯 Model CNN từ scratch | 📊 Độ chính xác cao | 🚀 Deployed on Streamlit Cloud</p>
        <p>Made with ❤️ for Vietnamese Cuisine</p>
    </div>
    """, unsafe_allow_html=True)

# Thêm thông báo
st.info("💡 **Lưu ý:** Lần đầu tiên chạy sẽ mất 10-15 phút để tải dataset và train model. Các lần sau sẽ nhanh hơn nhờ cache của Streamlit!")
