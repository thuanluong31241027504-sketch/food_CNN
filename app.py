import streamlit as st
import os
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Dense, Flatten, Dropout, BatchNormalization,
    GlobalAveragePooling2D, Input, Conv2D, MaxPooling2D
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import kagglehub
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="NHẬN DIỆN MÓN ĂN VIỆT NAM",
    page_icon="🍜",
    layout="wide"
)

# Constants
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 10
IMAGE_SIZE = (128, 128)

@st.cache_resource
def load_and_train():
    """Load dữ liệu và train model"""
    
    st.info("📥 Đang tải dữ liệu từ Kaggle (lần đầu mất 2-3 phút)...")
    path = kagglehub.dataset_download("quandang/vietnamese-foods")
    
    # Tạo dataframe
    def create_dataframe(directory):
        filepaths, labels = [], []
        if os.path.exists(directory):
            for label in os.listdir(directory):
                class_dir = os.path.join(directory, label)
                if os.path.isdir(class_dir):
                    for file in os.listdir(class_dir):
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            filepaths.append(os.path.join(class_dir, file))
                            labels.append(label)
        return pd.DataFrame({'filepath': filepaths, 'label': labels})
    
    train_df = create_dataframe(os.path.join(path, 'Train'))
    valid_df = create_dataframe(os.path.join(path, 'Validate'))
    
    if len(valid_df) == 0 and len(train_df) > 0:
        train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=42)
    
    st.success(f"✅ Train: {len(train_df)} | Validation: {len(valid_df)}")
    
    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
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
    
    class_names = list(train_generator.class_indices.keys())
    num_classes = len(class_names)
    
    # Model đơn giản
    st.info("🏗️ Đang xây dựng model...")
    model = Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(),
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Training
    st.info(f"🎓 Đang training model với {EPOCHS} epochs (mất 5-10 phút)...")
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    ]
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=valid_generator,
        callbacks=callbacks,
        verbose=0
    )
    
    st.success("✅ Training hoàn tất!")
    return model, class_names, history

def predict_image(image, model, class_names):
    """Dự đoán món ăn"""
    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 128, 128, 3)
    predictions = model.predict(img_array, verbose=0)[0]
    top_indices = np.argsort(predictions)[-3:][::-1]
    return [(class_names[i], predictions[i]) for i in top_indices]

# Main UI
st.title("🍜 NHẬN DIỆN MÓN ĂN VIỆT NAM")

# Load model
try:
    model, class_names, history = load_and_train()
    
    # Hiển thị accuracy
    acc = history.history['val_accuracy'][-1]
    st.success(f"✅ Model sẵn sàng! Độ chính xác: {acc:.2%}")
    
    # Upload ảnh
    uploaded_file = st.file_uploader("📤 Chọn ảnh món ăn...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        from PIL import Image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Ảnh của bạn", use_column_width=True)
        
        with col2:
            with st.spinner("🔍 Đang phân tích..."):
                results = predict_image(image, model, class_names)
            
            st.markdown("### 📊 Kết quả dự đoán:")
            for i, (food, prob) in enumerate(results, 1):
                if i == 1:
                    st.markdown(f"**🥇 {food}**")
                elif i == 2:
                    st.markdown(f"**🥈 {food}**")
                else:
                    st.markdown(f"**🥉 {food}**")
                st.progress(prob, text=f"Độ tin cậy: {prob:.1%}")
    
    # Danh sách món
    with st.expander("🍽️ Danh sách món ăn"):
        cols = st.columns(5)
        for i, food in enumerate(class_names):
            cols[i % 5].markdown(f"• {food}")

except Exception as e:
    st.error(f"Lỗi: {str(e)}")
    st.info("💡 Vui lòng đợi vài phút và thử lại. Lần đầu chạy cần thời gian để tải dữ liệu và train model.")
