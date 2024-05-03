import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode

st.set_option('deprecation.showfileUploaderEncoding', False)
showfileUploaderEncoding = False
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model(r'ResNet50_model.keras')
    return model
model = load_model()


def index_to_emotion(index):
    emotion_labels = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
    index_to_emotion = {index:emotion for emotion, index in emotion_labels.items()}
    emotion = index_to_emotion[index]
    return emotion


def model_prediction(test_image):
    
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(150,150))
    img_to_arr = tf.keras.preprocessing.image.img_to_array(image)
    
    img_arr = np.array([img_to_arr]) 
    img_arr = img_arr/255.0
    
    prediction = model.predict(img_arr)
    result_index = np.argmax(prediction)
    
    return result_index

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

class FaceEmotion(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        faces = face_cascade.detectMultiScale(image=img_rgb, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(x+w, y+h), color=(255, 0, 0), thickness=2)
            
            roi = img[y:y + h, x:x + w]
            
            
            roi_gray = cv2.resize(roi, (150, 150), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = tf.keras.preprocessing.image.img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                
                prediction = model.predict(roi)[0]
                result_index = int(np.argmax(prediction))
                output = index_to_emotion(result_index)
                
            
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

#sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Image Emotion Prediction", "Live emotion Prediction"])

# Home Page
if(app_mode == "Image Emotion Prediction"):
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
        
        
if(app_mode == "Live emotion Prediction"):
    st.header("Emotion Detection")
    
    
    st.title("Live Video Feed")
    webrtc_streamer(key="example", mode = WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory = FaceEmotion)
    
    
    
    
    
    
    


