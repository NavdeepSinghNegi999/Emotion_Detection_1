import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def index_to_emotion(index):
    emotion_labels = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
    index_to_emotion = {index:emotion for emotion, index in emotion_labels.items()}
    emotion = index_to_emotion[index]
    return emotion


def model_prediction(test_image):
    model = tf.keras.models.load_model(r'model/ResNet50_model/ResNet50_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(150,150))
    img_to_arr = tf.keras.preprocessing.image.img_to_array(image)
    
    img_arr = np.array([img_to_arr]) 
    img_arr = img_arr/255.0
    
    prediction = model.predict(img_arr)
    result_index = np.argmax(prediction)
    
    return result_index


# #sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Emotion Prediction Image"])

# Home Page
if(app_mode == "Emotion Prediction Image"):
    st.header("Emotion Detection")
    
    # Load image
    test_image = st.file_uploader("Choose an Image")
    
    if(st.button("Show Image")):
        st.image(test_image, use_column_width=True)
    
    if(st.button("Predict")):
        st.write("Our Prediction")
        result_index  = model_prediction(test_image)
        result = index_to_emotion(result_index)
        st.success(f"{result}")
    
    
    
    
    


