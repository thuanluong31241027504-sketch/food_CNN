import streamlit as st
import sys

st.title("Test Python Version")
st.write(f"Python version: {sys.version}")

try:
    import tensorflow as tf
    st.success(f"✅ TensorFlow version: {tf.__version__}")
    st.success("TensorFlow loaded successfully!")
except Exception as e:
    st.error(f"❌ TensorFlow error: {e}")
