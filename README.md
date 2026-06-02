# 🍜 Vietnamese Food Recognition

Ứng dụng nhận diện món ăn Việt Nam sử dụng CNN với kiến trúc đã được train thành công trên Google Colab.

## 🏗️ Kiến trúc Model (Giữ nguyên từ Colab)

```python
# 4 Convolutional Blocks
- Block 1: 2x Conv2D(32) + BatchNorm + MaxPool + Dropout(0.25)
- Block 2: 2x Conv2D(64) + BatchNorm + MaxPool + Dropout(0.30)
- Block 3: 2x Conv2D(128) + BatchNorm + MaxPool + Dropout(0.35)
- Block 4: Conv2D(256) + BatchNorm + MaxPool + Dropout(0.40)

# Head
- GlobalAveragePooling2D
- Dense(256) + Dropout(0.5)
- Dense(30, softmax)
