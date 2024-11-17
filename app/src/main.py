import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import os
import sqlite3
from datetime import datetime
import hashlib
import base64

# Database setup and authentication functions
def init_db():
    conn = sqlite3.connect('music_genre_app.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  email TEXT UNIQUE NOT NULL)''')
    
    # Create predictions table
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  prediction TEXT NOT NULL,
                  confidence REAL NOT NULL,
                  filename TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def register_user(username, password, email):
    conn = sqlite3.connect('music_genre_app.db')
    c = conn.cursor()
    try:
        hashed_pw = hash_password(password)
        c.execute('INSERT INTO users (username, password, email) VALUES (?, ?, ?)',
                 (username, hashed_pw, email))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect('music_genre_app.db')
    c = conn.cursor()
    hashed_pw = hash_password(password)
    c.execute('SELECT id, username FROM users WHERE username = ? AND password = ?',
             (username, hashed_pw))
    user = c.fetchone()
    conn.close()
    return user

def save_prediction(user_id, prediction, confidence, filename):
    conn = sqlite3.connect('music_genre_app.db')
    c = conn.cursor()
    c.execute('INSERT INTO predictions (user_id, prediction, confidence, filename) VALUES (?, ?, ?, ?)',
             (user_id, prediction, confidence, filename))
    conn.commit()
    conn.close()

def get_user_predictions(user_id):
    conn = sqlite3.connect('music_genre_app.db')
    c = conn.cursor()
    c.execute('''SELECT prediction, confidence, filename, timestamp 
                 FROM predictions WHERE user_id = ? 
                 ORDER BY timestamp DESC''', (user_id,))
    predictions = c.fetchall()
    conn.close()
    return predictions

# Initialize database
init_db()

# Function to load the trained model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("app/src/Trained_model.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load and preprocess audio data
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    
    # Normalize the audio data
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Perform preprocessing
    chunk_duration = 4
    overlap_duration = 2
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
    
    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)
    
    return np.array(data)

# TensorFlow Model Prediction
def model_prediction(X_test):
    try:
        model = load_model()
        if model is None:
            return None, None
            
        predictions = model.predict(X_test)
        avg_predictions = np.mean(predictions, axis=0)
        predicted_class = np.argmax(avg_predictions)
        confidence_scores = avg_predictions * 100
        
        return predicted_class, confidence_scores
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# Function to record audio
def record_audio(duration, sample_rate=22050):
    try:
        st.info(f"Recording for {duration} seconds...")
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        write(temp_file.name, sample_rate, (audio_data * 32767).astype(np.int16))
        st.info(f"Recording saved to {temp_file.name}")
        
        return temp_file.name
    
    except Exception as e:
        st.error(f"Error recording audio: {str(e)}")
        return None

# Session state initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None

# Set page config
st.set_page_config(page_title="Music Genre Classification", layout="wide")

# Custom CSS with red theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Orbitron', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0a0a0a, #1a1a1a);
        color: #e0e0e0;
    }

    .stButton > button {
        background-color: #ff0033;
        color: white;
        border-radius: 20px;
        font-weight: bold;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 0 15px rgba(255, 0, 51, 0.5);
        border: none;
    }

    .stButton > button:hover {
        background-color: #ff3366;
        box-shadow: 0 0 25px rgba(255, 51, 102, 0.7);
        transform: translateY(-2px);
    }

    .stRadio > div {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 10px;
        backdrop-filter: blur(5px);
    }

    .stSlider > div > div {
        background-color: #ff0033;
    }

    .card {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 0, 51, 0.2);
        box-shadow: 0 0 20px rgba(255, 0, 51, 0.2);
        transition: all 0.3s ease;
    }

    .card:hover {
        box-shadow: 0 0 30px rgba(255, 0, 51, 0.4);
        transform: translateY(-5px);
    }

    .prediction-result {
        font-size: 2.5em;
        color: #ff3366;
        text-align: center;
        padding: 20px;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(255, 51, 102, 0.7);
    }

    .confidence-bar {
        height: 20px;
        background-color: #ff0033;
        border-radius: 10px;
        transition: width 0.5s ease-in-out;
        box-shadow: 0 0 10px rgba(255, 0, 51, 0.5);
    }

    h1, h2, h3 {
        color: #ff3366;
        text-shadow: 0 0 10px rgba(255, 51, 102, 0.5);
    }

    .rotating-icon {
        display: inline-block;
        animation: rotate 5s linear infinite;
    }

    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    .genre-list {
        columns: 2;
        -webkit-columns: 2;
        -moz-columns: 2;
    }

    .genre-item {
        margin-bottom: 10px;
        transition: transform 0.3s ease;
    }

    .genre-item:hover {
        transform: translateX(10px);
        color: #ff3366;
    }

    .top-bar {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 60px;
        background-color: rgba(0, 0, 0, 0.7);
        backdrop-filter: blur(10px);
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 20px;
        z-index: 1000;
    }

    .top-bar-title {
        color: #ff3366;
        font-size: 1.5em;
        font-weight: bold;
    }

    .top-bar-menu {
        display: flex;
        gap: 20px;
    }

    .top-bar-menu a {
        color: #e0e0e0;
        text-decoration: none;
        transition: color 0.3s ease;
    }

    .top-bar-menu a:hover {
        color: #ff3366;
    }

    .content {
        margin-top: 80px;
        padding: 20px;
    }

    /* Login form styles */
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 20px;
    }

    .login-form {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }

    .form-input {
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Authentication UI
def show_login_page():
    st.markdown("<h1 class='prediction-result'>Login</h1>", unsafe_allow_html=True)
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            user = login_user(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.user_id = user[0]
                st.session_state.username = user[1]
                st.success("Successfully logged in!")
                
            else:
                st.error("Invalid username or password")
    
    st.markdown("---")
    st.markdown("Don't have an account? Register below!")
    
    with st.form("register_form"):
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        email = st.text_input("Email")
        register = st.form_submit_button("Register")
        
        if register:
            if register_user(new_username, new_password, email):
                st.success("Registration successful! Please login.")
            else:
                st.error("Username or email already exists")

# History page
def show_history_page():
    st.markdown("<h1 class='prediction-result'>Prediction History</h1>", unsafe_allow_html=True)
    
    predictions = get_user_predictions(st.session_state.user_id)
    
    if not predictions:
        st.info("No predictions yet!")
    else:
        for pred in predictions:
            with st.expander(f"{pred[0]} - {pred[3]}", expanded=False):
                st.markdown(f"""
                <div class='card'>
                    <p><strong>Genre:</strong> {pred[0]}</p>
                    <p><strong>Confidence:</strong> {pred[1]:.2f}%</p>
                    <p><strong>File:</strong> {pred[2]}</p>
                    <p><strong>Date:</strong> {pred[3]}</p>
                </div>
                """, unsafe_allow_html=True)

# Top navigation bar
st.markdown("""
<div class="top-bar">
    <div class="top-bar-title">
        <img src="logo.png" alt="Logo" style="height: 40px; vertical-align: middle; margin-right: 10px;">
        Music Genre Classification
    </div>
    <div class="top-bar-menu">
        <a href="javascript:void(0);" onclick="changePage('Home')">Home</a>
        <a href="javascript:void(0);" onclick="changePage('About')">About</a>
        <a href="javascript:void(0);" onclick="changePage('Prediction')">Prediction</a>
        <a href="javascript:void(0);" onclick="changePage('History')">History</a>
    </div>
</div>
""", unsafe_allow_html=True)

# Main content
st.markdown('<div class="content">', unsafe_allow_html=True)

# Navigation based on login status
if not st.session_state.logged_in:
    show_login_page()
else:
    # Add logout button in the sidebar
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.username = None
        
    
    st.sidebar.markdown(f"Welcome, {st.session_state.username}!")
    
    # Navigation
    app_mode = st.selectbox("Select Page", 
                           ["Home", "About", "Prediction", "History"],
                           key="navigation",
                           label_visibility="hidden")
    
    if app_mode == "Home":
        st.markdown("<h1 class='prediction-result'>Welcome to Music Genre Classification <span class='rotating-icon'>🎶</span></h1>", unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
            <p style="font-size: 1.2em;">Upload or record an audio file to predict its genre. Our system classifies music into 10 genres efficiently.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h3>Goal</h3>
            <p>Automatically classify music into one of 10 genres:</p>
            <div class="genre-list">
                <div class="genre-item">🎷 Blues</div>
                <div class="genre-item">🎻 Classical</div>
                <div class="genre-item">🤠 Country</div>
                <div class="genre-item">💃 Disco</div>
                <div class="genre-item">🎤 Hip Hop</div>
                <div class="genre-item">🎺 Jazz</div>
                <div class="genre-item">🤘 Metal</div>
                <div class="genre-item">🎵 Pop</div>
                <div class="genre-item">🌴 Reggae</div>
                <div class="genre-item">🎸 Rock</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    elif app_mode == "About":
        st.markdown("<h1 class='prediction-result'>About the Project <span class='rotating-icon'>🎵</span></h1>", unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
            <p style="font-size: 1.2em;">This project uses deep learning to classify music into 10 genres.</p>
            <h3>Technology Stack:</h3>
            <ul>
                <li>TensorFlow for deep learning</li>
                <li>Streamlit for the web interface</li>
                <li>Librosa for audio processing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    elif app_mode == "Prediction":
        st.markdown("<h1 class='prediction-result'>Model Prediction <span class='rotating-icon'>🎼</span></h1>", unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
            <p style="font-size: 1.2em;">Choose whether to upload or record an audio file, and get the predicted genre along with confidence scores.</p>
        </div>
        """, unsafe_allow_html=True)

        # Option to upload or record audio
        input_type = st.radio("Select Input Type:", ("Upload Audio File", "Record Audio"))
        filepath = None

        if input_type == "Upload Audio File":
            test_audio = st.file_uploader("Upload an audio file", type=["mp3", "wav"], key="upload")
            if test_audio is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                    temp_file.write(test_audio.read())
                    filepath = temp_file.name
                st.audio(filepath)
                
        elif input_type == "Record Audio":
            col1, col2 = st.columns(2)
            with col1:
                duration = st.slider("Recording duration (seconds):", 1, 30, 15)
            with col2:
                if st.button("Start Recording"):
                    filepath = record_audio(duration)
                    if filepath:
                        st.audio(filepath)
                        st.session_state.recorded_file = filepath
                        st.success("Recording saved successfully!")

        if filepath:
            st.session_state.filepath = filepath

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
                            
                            # Show prediction result
                            st.balloons()
                            st.markdown(f"<div class='card prediction-result'>Prediction: {labels[result_index].upper()}</div>", unsafe_allow_html=True)
                            
                            # Display confidence scores
                            st.markdown("<div class='card'>", unsafe_allow_html=True)
                            st.markdown("### Confidence Scores", unsafe_allow_html=True)
                            top_3_indices = np.argsort(confidence_scores)[-3:][::-1]
                            for idx in top_3_indices:
                                confidence = confidence_scores[idx]
                                st.markdown(f"<div class='genre-item'><strong>{labels[idx].title()}:</strong> {confidence:.2f}%</div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='confidence-bar' style='width: {confidence}%;'></div>", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Save prediction to database
                            save_prediction(
                                st.session_state.user_id,
                                labels[result_index],
                                confidence_scores[result_index],
                                os.path.basename(st.session_state.filepath)
                            )
    
    elif app_mode == "History":
        show_history_page()

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown(f"""
<div style="position: fixed; left: 0; bottom: 0; width: 100%; background-color: rgba(0, 0, 0, 0.7); color: white; text-align: center; padding: 10px;">
    Developed Hassan Muhammad Yousuf | {st.session_state.username if st.session_state.logged_in else 'Please login'}
</div>
""", unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == "__main__":
    pass