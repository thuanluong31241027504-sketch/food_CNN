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

# Cấu hình trang
st.set_page_config(
    page_title="NHẬN DIỆN MÓN ĂN VIỆT NAM",
    page_icon="🍜",
    layout="wide"
)

# Constants (GIỮ NGUYÊN từ code Colab)
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 15
IMAGE_SIZE = (128, 128)
BASE_DIRS = []

# Hàm tạo dataframe (GIỮ NGUYÊN)
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
    """Load dữ liệu và train model"""
    
    with st.spinner("📥 Đang tải dữ liệu từ Kaggle..."):
        path = kagglehub.dataset_download("quandang/vietnamese-foods")
        BASE_DIRS.append(path)
        
        train_df = merge_datasets(BASE_DIRS, 'Train')
        valid_df = merge_datasets(BASE_DIRS, 'Validate')
        test_df = merge_datasets(BASE_DIRS, 'Test')
        
        if len(valid_df) == 0 and len(train_df) > 0:
            train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=42)
    
    st.success(f"✅ Train: {len(train_df)} | Validate: {len(valid_df)} | Test: {len(test_df)}")
    
    # Data Augmentation
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
    
    train_generator = train_datagen.flow_from_dataframe(
        train_df, x_col='filepath', y_col='label', 
        target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
    )
    valid_generator = valid_datagen.flow_from_dataframe(
        valid_df, x_col='filepath', y_col='label', 
        target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
    )
    
    CLASS_NAMES = sorted(train_generator.class_indices.keys())
    num_classes = len(CLASS_NAMES)
    
    # Xây dựng model (GIỮ NGUYÊN cấu trúc)
    with st.spinner("🏗️ Đang xây dựng model CNN..."):
        model = Sequential([
            Input(shape=(128, 128, 3)),
            
            Conv2D(32, (3,3), padding='same', activation='relu'),
            BatchNormalization(),
            Conv2D(32, (3,3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(),
            Dropout(0.25),
            
            Conv2D(64, (3,3), padding='same', activation='relu'),
            BatchNormalization(),
            Conv2D(64, (3,3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(),
            Dropout(0.30),
            
            Conv2D(128, (3,3), padding='same', activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3,3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(),
            Dropout(0.35),
            
            Conv2D(256, (3,3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(),
            Dropout(0.40),
            
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
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-7, verbose=0)
    ]
    
    # Training
    with st.spinner(f"🎓 Đang training model với {EPOCHS} epochs (mất 5-10 phút)..."):
        history = model.fit(
            train_generator, epochs=EPOCHS, 
            validation_data=valid_generator, callbacks=callbacks, verbose=0
        )
    
    st.success("✅ Training hoàn tất!")
    return model, CLASS_NAMES, history

def predict_image(image, model, class_names):
    """Dự đoán món ăn"""
    from PIL import Image as PILImage
    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 128, 128, 3)
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    return class_names[predicted_class], confidence

# Main UI
st.title("🍜 NHẬN DIỆN MÓN ĂN VIỆT NAM")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1999/1999625.png", width=100)
    st.markdown("### 📌 Thông tin")
    st.markdown(f"- Batch Size: {BATCH_SIZE}")
    st.markdown(f"- Epochs: {EPOCHS}")
    st.markdown(f"- Input size: 128x128")
    if st.button("🔄 Train lại model", type="primary"):
        st.cache_resource.clear()
        st.rerun()

# Load model
model, class_names, history = load_and_train()

# Hiển thị kết quả
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🎯 Final Accuracy", f"{history.history['val_accuracy'][-1]:.2%}")
with col2:
    st.metric("📈 Best Accuracy", f"{max(history.history['val_accuracy']):.2%}")
with col3:
    st.metric("🔄 Epochs", len(history.history['val_accuracy']))

# Upload ảnh
uploaded_file = st.file_uploader("📤 Chọn ảnh món ăn...", type=['jpg', 'jpeg', 'png', 'webp'])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    with col1:
        image = plt.imread(uploaded_file)
        st.image(image, caption="Ảnh đã upload", use_column_width=True)
    with col2:
        with st.spinner("🔍 Đang phân tích..."):
            from PIL import Image as PILImage
            pil_image = PILImage.open(uploaded_file)
            food_name, confidence = predict_image(pil_image, model, class_names)
        
        st.markdown(f"""
        <div style="
            padding: 20px;
            border-radius: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-align: center;
        ">
            <h1>🍽️ {food_name}</h1>
            <p>Độ tin cậy: {confidence:.1%}</p>
        </div>
        """, unsafe_allow_html=True)

# Danh sách món ăn
with st.expander("🍽️ Danh sách 30 món ăn"):
    cols = st.columns(5)
    for idx, food in enumerate(class_names):
        cols[idx % 5].markdown(f"• {food}")

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Powered by TensorFlow & Streamlit</div>", unsafe_allow_html=True)
