import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import time
import os

# Function to load the trained model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("Trained_model.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load and preprocess audio data
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    
    # Normalize the audio data (same normalization as during training)
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Perform preprocessing (convert to Mel spectrogram and resize)
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
    
    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        
        # Compute Mel spectrogram for the chunk
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)
    
    return np.array(data)

# TensorFlow Model Prediction
def model_prediction(X_test):
    try:
        model = load_model()
        if model is None:
            return None
            
        # Make prediction
        predictions = model.predict(X_test)
        
        # Average predictions across chunks
        avg_predictions = np.mean(predictions, axis=0)
        
        # Get the predicted class
        predicted_class = np.argmax(avg_predictions)
        
        # Get confidence scores
        confidence_scores = avg_predictions * 100
        
        return predicted_class, confidence_scores
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# Function to record audio
def record_audio(duration, sample_rate=22050):
    try:
        # Record the audio using sounddevice
        st.info(f"Recording for {duration} seconds...")
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()  # Wait for the recording to finish
        
        # Normalize audio data (ensure consistent scaling with model input)
        audio_data = audio_data / np.max(np.abs(audio_data))  # Normalize to -1 to 1
        
        # Save the recorded audio to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        write(temp_file.name, sample_rate, (audio_data * 32767).astype(np.int16))  # Convert to 16-bit PCM
        st.info(f"Recording saved to {temp_file.name}")
        
        return temp_file.name  # Return the file path
    
    except Exception as e:
        st.error(f"Error recording audio: {str(e)}")
        return None

# Streamlit UI
st.set_page_config(page_title="Music Genre Classification", layout="wide")

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

# Home Page
if app_mode == "Home":
    st.markdown(
        """
        <style>
        .stApp { background-color: #181646; color: white; }
        h2, h3 { color: white; }
        </style>
        """, unsafe_allow_html=True
    )
    st.markdown("## Welcome to the Music Genre Classification System! 🎶🎧")
    if os.path.exists("music_genre_home.png"):
        st.image("music_genre_home.png", use_column_width=True)
    st.markdown("""
        **Our goal is to help identify music genres from audio tracks efficiently.**
        **Upload or record an audio file, and let our system detect its genre.**
    """)

# About Project Page
elif app_mode == "About Project":
    st.markdown("""
        ### About Project
        This project classifies music into 10 genres using deep learning.
        
        #### Genres Classified:
        - Blues
        - Classical
        - Country
        - Disco
        - Hip Hop
        - Jazz
        - Metal
        - Pop
        - Reggae
        - Rock
    """)

# Prediction Page
elif app_mode == "Prediction":
    st.header("Model Prediction")
    
    # Option to upload or record audio
    input_type = st.radio("Select Input Type:", ("Upload Audio File", "Record Audio"))
    filepath = None

    if input_type == "Upload Audio File":
        test_audio = st.file_uploader("Upload an audio file", type=["mp3", "wav"])
        if test_audio is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(test_audio.read())
                filepath = temp_file.name
            st.audio(filepath)
            
    elif input_type == "Record Audio":
        col1, col2 = st.columns(2)
        with col1:
            duration = st.slider("Recording duration (seconds):", 1, 10, 5)
        with col2:
            if st.button("Start Recording"):
                filepath = record_audio(duration)  # Use the new record_audio function
                if filepath:
                    st.audio(filepath)
                    st.session_state.recorded_file = filepath
                    st.success("Recording saved successfully!")

    # Store filepath in session state
    if filepath:
        st.session_state.filepath = filepath

    # Predict Button
    if st.button("Predict"):
        if not hasattr(st.session_state, 'filepath'):
            st.warning("Please record or upload an audio file first.")
        else:
            with st.spinner("Analyzing audio..."):
                # Load and preprocess the audio
                X_test = load_and_preprocess_data(st.session_state.filepath)
                
                if X_test is not None:
                    # Get prediction and confidence scores
                    result_index, confidence_scores = model_prediction(X_test)
                    
                    if result_index is not None:
                        labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 
                                  'jazz', 'metal', 'pop', 'reggae', 'rock']
                        
                        # Show main prediction
                        st.balloons()
                        st.markdown(f"## Prediction: :red[{labels[result_index].upper()}]")
                        
                        # Show confidence scores
                        st.markdown("### Confidence Scores:")
                        
                        # Sort confidence scores and display top 3
                        top_3_indices = np.argsort(confidence_scores)[-3:][::-1]
                        for idx in top_3_indices:
                            confidence = confidence_scores[idx]
                            st.markdown(f"**{labels[idx].title()}:** {confidence:.2f}%")
