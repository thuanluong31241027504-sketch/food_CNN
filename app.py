import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import kagglehub
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

# Cấu hình trang
st.set_page_config(
    page_title="Nhận diện món ăn Việt Nam 🍜",
    page_icon="🍜",
    layout="wide"
)

# Constants
BATCH_SIZE = 16  # Giảm để tránh memory error
LEARNING_RATE = 0.001
EPOCHS = 10  # Giảm để train nhanh
IMAGE_SIZE = (128, 128)

@st.cache_resource
def load_and_train():
    """Load dataset và train model"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Download dataset
    status_text.text("📥 Đang tải dữ liệu từ Kaggle...")
    progress_bar.progress(10)
    
    path = kagglehub.dataset_download("quandang/vietnamese-foods")
    progress_bar.progress(20)
    
    # Tạo dataframe
    def create_df(directory):
        filepaths, labels = [], []
        if os.path.exists(directory):
            for label in os.listdir(directory):
                class_dir = os.path.join(directory, label)
                if os.path.isdir(class_dir):
                    for file in os.listdir(class_dir):
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            filepaths.append(os.path.join(class_dir, file))
                            labels.append(label)
        return filepaths, labels
    
    # Load data
    train_paths, train_labels = create_df(os.path.join(path, 'Train'))
    valid_paths, valid_labels = create_df(os.path.join(path, 'Validate'))
    
    # Nếu không có validation, split từ train
    if len(valid_paths) == 0 and len(train_paths) > 0:
        from sklearn.model_selection import train_test_split
        train_paths, valid_paths, train_labels, valid_labels = train_test_split(
            train_paths, train_labels, test_size=0.2, random_state=42
        )
    
    progress_bar.progress(30)
    status_text.text(f"✅ Đã tải: {len(train_paths)} ảnh train, {len(valid_paths)} ảnh validation")
    
    # Data augmentation
    status_text.text("🔄 Đang chuẩn bị data...")
    progress_bar.progress(40)
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    
    valid_datagen = ImageDataGenerator(rescale=1./255)
    
    # Tạo dataframe cho flow_from_dataframe
    import pandas as pd
    train_df = pd.DataFrame({'filepath': train_paths, 'label': train_labels})
    valid_df = pd.DataFrame({'filepath': valid_paths, 'label': valid_labels})
    
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
    
    progress_bar.progress(50)
    
    # Xây dựng model đơn giản
    status_text.text("🏗️ Đang xây dựng model...")
    
    model = Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),
        Conv2D(32, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(128, (3,3), activation='relu'),
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
    
    progress_bar.progress(60)
    
    # Training
    status_text.text(f"🎓 Đang training model... (có thể mất 5-10 phút)")
    
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
    
    progress_bar.progress(100)
    status_text.text("✅ Training hoàn tất!")
    
    return model, class_names, history

def predict_image(image, model, class_names):
    """Dự đoán ảnh"""
    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 128, 128, 3)
    
    predictions = model.predict(img_array, verbose=0)[0]
    top_indices = np.argsort(predictions)[-3:][::-1]
    
    results = [(class_names[idx], predictions[idx]) for idx in top_indices]
    return results

# Main UI
st.title("🍜 Nhận diện món ăn Việt Nam")
st.markdown("Upload ảnh món ăn để nhận diện")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1999/1999625.png", width=100)
    st.markdown("### Hướng dẫn")
    st.markdown("1. Chờ model khởi tạo (lần đầu 5-10 phút)")
    st.markdown("2. Upload ảnh món ăn")
    st.markdown("3. Xem kết quả dự đoán")
    
    if st.button("🔄 Train lại model"):
        st.cache_resource.clear()
        st.rerun()

# Load model
try:
    model, class_names, history = load_and_train()
    
    # Hiển thị accuracy
    acc = history.history['val_accuracy'][-1]
    st.success(f"✅ Model sẵn sàng! Độ chính xác: {acc:.2%}")
    
    # Upload ảnh
    uploaded_file = st.file_uploader("Chọn ảnh...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Ảnh của bạn", use_column_width=True)
        
        with col2:
            with st.spinner("Đang phân tích..."):
                results = predict_image(image, model, class_names)
            
            st.markdown("### Kết quả:")
            for i, (food, prob) in enumerate(results, 1):
                st.markdown(f"**{i}. {food}**")
                st.progress(prob, text=f"Độ tin cậy: {prob:.1%}")
    
    # Hiển thị danh sách món
    with st.expander("Danh sách món ăn"):
        cols = st.columns(5)
        for i, food in enumerate(class_names):
            cols[i % 5].markdown(f"- {food}")

except Exception as e:
    st.error(f"Lỗi: {str(e)}")
    st.markdown("""
    ### Cách khắc phục:
    1. Kiểm tra kết nối internet
    2. Refresh trang và thử lại
    3. Nếu vẫn lỗi, đợi vài phút rồi thử lại
    """)
