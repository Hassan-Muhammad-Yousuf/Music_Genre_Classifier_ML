import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize
from scipy.io.wavfile import write
import tempfile
import os
import sqlite3
from datetime import datetime
import hashlib
import base64
from pathlib import Path

# Set page config as the first Streamlit command
st.set_page_config(page_title="Music Genre Classification", layout="wide")

# Initialize the database
def init_db():
    conn = sqlite3.connect('music_genre_app.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL
        )
    ''')
    
    # Create predictions table with a foreign key linking to users
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            filename TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    conn.close()

# Update database structure
def update_db_structure():
    conn = sqlite3.connect('music_genre_app.db')
    c = conn.cursor()
    try:
        c.execute("ALTER TABLE predictions ADD COLUMN user_id INTEGER NOT NULL DEFAULT 0")
        conn.commit()
    except sqlite3.OperationalError:
        # Column already exists
        pass
    finally:
        conn.close()

# Initialize and update the database
init_db()
update_db_structure()

# Hash passwords securely
def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

# Register a new user
def register_user(username, password, email):
    conn = sqlite3.connect('music_genre_app.db')
    c = conn.cursor()
    try:
        hashed_pw = hash_password(password)
        c.execute(
            'INSERT INTO users (username, password, email) VALUES (?, ?, ?)',
            (username, hashed_pw, email)
        )
        conn.commit()
        return True  # Registration successful
    except sqlite3.IntegrityError:
        return False  # Username or email already exists
    finally:
        conn.close()

# Authenticate user during login
def login_user(username, password):
    conn = sqlite3.connect('music_genre_app.db')
    c = conn.cursor()
    hashed_pw = hash_password(password)
    c.execute(
        'SELECT id, username FROM users WHERE username = ? AND password = ?',
        (username, hashed_pw)
    )
    user = c.fetchone()
    conn.close()
    return user  # Returns (id, username) if successful, None otherwise

# Save a prediction for a user
def save_prediction(user_id, prediction, confidence, filename):
    conn = sqlite3.connect('music_genre_app.db')
    c = conn.cursor()
    try:
        c.execute(
            'INSERT INTO predictions (user_id, prediction, confidence, filename) VALUES (?, ?, ?, ?)',
            (user_id, prediction, confidence, filename)
        )
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error saving prediction: {e}")
    finally:
        conn.close()

# Retrieve prediction history for a user
def get_user_predictions(user_id):
    conn = sqlite3.connect('music_genre_app.db')
    c = conn.cursor()
    try:
        c.execute('''
            SELECT prediction, confidence, filename, timestamp 
            FROM predictions 
            WHERE user_id = ? 
            ORDER BY timestamp DESC
        ''', (user_id,))
        predictions = c.fetchall()
    except sqlite3.Error as e:
        print(f"Error retrieving predictions: {e}")
        predictions = []
    finally:
        conn.close()
    return predictions

# Function to load the trained model
@st.cache_resource
def load_model():
    try:
        # Determine the model path dynamically
        current_dir = Path(__file__).resolve().parent  # Get the current script's directory
        model_path = current_dir / 'Trained_model.keras'  # Construct the model's full path

        # Check if the model file exists
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure it exists in the root directory.")

        # Load the model
        model = tf.keras.models.load_model(model_path)
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
# def record_audio(duration, sample_rate=22050):
#     try:
#         st.info(f"Recording for {duration} seconds...")
#         sd.default.device = None  # Use default audio device
#         audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32', device=None)
#         sd.wait()
#         
#         # Normalize the audio data
#         audio_data = audio_data.flatten()  # Flatten the array
#         audio_data = audio_data / np.max(np.abs(audio_data))
#         
#         temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
#         write(temp_file.name, sample_rate, (audio_data * 32767).astype(np.int16))
#         st.info(f"Recording saved to {temp_file.name}")
#         
#         return temp_file.name
#     
#     except Exception as e:
#         st.error(f"Error recording audio: {str(e)}")
#         return None

# Session state initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None

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
    st.markdown("""
    <h1 class='prediction-result'>
        Music Genre Classifier<br>Login
    </h1>
    """, unsafe_allow_html=True)    
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
                st.rerun()
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
            # Safely handle each prediction field
            try:
                # Genre
                genre = str(pred[0]) if pred[0] is not None else "Unknown"
                
                # Confidence - additional safety checks
                try:
                    # Handle potential byte or string conversion
                    if isinstance(pred[1], bytes):
                        confidence = "N/A"
                    else:
                        confidence = f"{float(str(pred[1])):.2f}%" if pred[1] is not None else "N/A"
                except (ValueError, TypeError):
                    confidence = "N/A"
                
                # Filename
                filename = str(pred[2]) if pred[2] is not None else "No file"
                
                # Timestamp
                timestamp = str(pred[3]) if pred[3] is not None else "Unknown date"
                
                # Create expander
                with st.expander(f"{genre} - {timestamp}", expanded=False):
                    history_details = f"""
                    **Genre:** {genre}

                    **Confidence:** {confidence}

                    **File:** {filename}

                    **Date:** {timestamp}
                    """
                    st.markdown(history_details)
            
            except Exception as e:
                # Log the error and continue processing other predictions
                st.error(f"Error processing prediction: {e}")
                continue

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
        st.rerun()
    
    st.sidebar.markdown(f"""
🎶 **Welcome, {st.session_state.username}!** 🎵  

Step into a world where every note tells a story and every rhythm finds its identity. 🌟  

Here at **Genre Vibes**, we believe that music isn't just sound—it's an expression of who we are.  

With our genre classifier, you're about to unlock the magic behind your favorite tunes.  
From the soulful depths of blues, the electrifying energy of rock, the poetic vibes of hip-hop,  
to the serene flow of classical melodies—we're here to help you explore the essence of sound like never before.  

✨ **Upload or Record your music**  
✨ **Discover its genre**  
✨ **Expand your playlist**  

Your journey into the diverse world of music genres starts now.  
Thank you for making us a part of your musical adventure! 🎧  

Let's redefine how you experience music, one song at a time! 💫
""")

    
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
        <p>Explore the fascinating history of music, where every beat and rhythm tells a unique story. From the soulful origins that shaped emotional expression to the energetic sounds that define the spirit of rebellion, music has always been a powerful force for change. Dive deep into the roots of these musical movements, understand the evolution that has shaped them, and discover how they continue to inspire and influence artists and fans worldwide. The journey through these sounds is a journey through culture, history, and the heartbeat of generations.</p>
    </div>

    <div class="card">
        <div class="genre-item">
            <h4>🎷 <b>Blues:</b></h4>
            <p>The blues originated in the deep South of the United States around the end of the 19th century. It was influenced by African American spirituals, work songs, and folk music. Blues has been a major influence on many other genres, including jazz, rock, and R&B.</p>
        </div>
        <div class="genre-item">
            <h4>🎻 <b>Classical:</b></h4>
            <p>Classical music is a broad term that refers to a long tradition of music that spans over a thousand years. Its roots go back to the Western church music of the medieval period, with significant contributions from composers like Beethoven, Mozart, and Bach. It's known for its complex structures and orchestral compositions.</p>
        </div>
        <div class="genre-item">
            <h4>🤠 <b>Country:</b></h4>
            <p>Country music originated in the rural Southern United States in the 1920s. It blends elements of folk, Western, and blues music. The genre is known for its twangy guitars and storytelling lyrics. Artists like Johnny Cash and Dolly Parton have made it globally popular.</p>
        </div>
        <div class="genre-item">
            <h4>💃 <b>Disco:</b></h4>
            <p>Disco emerged in the 1970s, primarily in urban nightlife scenes. It combines elements of funk, soul, pop, and dance music, and was characterized by its upbeat rhythms and glamorous, danceable beats. It became extremely popular in clubs and influenced dance music for decades.</p>
        </div>
        <div class="genre-item">
            <h4>🎤 <b>Hip Hop:</b></h4>
            <p>Hip hop culture originated in the Bronx, New York City, during the late 1970s. Initially, it consisted of four elements: rapping, DJing, graffiti, and breakdancing. Hip hop has become one of the most influential genres, representing youth culture and political expression through music, fashion, and art.</p>
        </div>
        <div class="genre-item">
            <h4>🎺 <b>Jazz:</b></h4>
            <p>Jazz originated in the early 20th century in New Orleans, Louisiana. It's characterized by swing and blue notes, call and response vocals, and improvisation. It is heavily influenced by African American musical traditions, and artists like Louis Armstrong and Duke Ellington helped shape its development.</p>
        </div>
        <div class="genre-item">
            <h4>🤘 <b>Metal:</b></h4>
            <p>Heavy metal emerged in the late 1960s and early 1970s, influenced by hard rock, blues, and psychedelia. It is known for its loud, aggressive sound, distorted guitars, and fast tempos. Bands like Black Sabbath and Metallica helped define the genre, which has spawned numerous subgenres.</p>
        </div>
        <div class="genre-item">
            <h4>🎵 <b>Pop:</b></h4>
            <p>Pop music is a genre that is characterized by its catchy melodies, upbeat tempos, and broad appeal. It emerged in the 1950s and quickly became the dominant genre worldwide, thanks to artists like The Beatles, Michael Jackson, and Madonna. Pop music often blends elements from various other genres.</p>
        </div>
        <div class="genre-item">
            <h4>🌴 <b>Reggae:</b></h4>
            <p>Reggae music originated in Jamaica in the late 1960s. It developed from earlier genres like ska and rocksteady, characterized by rhythmic accents on the offbeat and socially conscious lyrics. Bob Marley, one of the genre's most famous icons, used reggae as a vehicle for political and social change.</p>
        </div>
        <div class="genre-item">
            <h4>🎸 <b>Rock:</b></h4>
            <p>Rock music began in the 1950s, heavily influenced by rhythm and blues, jazz, and gospel. Its early pioneers, like Chuck Berry and Elvis Presley, shaped the genre, which evolved through the decades into many subgenres. Rock remains one of the most popular and diverse genres globally.</p>
        </div>
    </div>
""", unsafe_allow_html=True)
    
    elif app_mode == "About":
        st.markdown("<h1 class='prediction-result'>About the Project <span class='rotating-icon'>🎵</span></h1>", unsafe_allow_html=True)
        st.markdown("""
            <p style="font-size: 1.2em;">Welcome to the **Music Genre Classifier** project! This application is designed to explore the fascinating world of music genres using the power of **Deep Learning**. By analyzing the unique characteristics of audio signals, it can accurately predict the genre of your favorite tracks.</p>

            <h2>Why This Project?</h2>
            <p style="font-size: 1.1em;">Music is universal, but its genres provide a deeper understanding of rhythm, instruments, and culture. This project aims to bridge the gap between human intuition and machine intelligence in identifying musical genres. Whether you're an artist, a music enthusiast, or a developer curious about AI in audio, this app has something for you!</p>

            <h2>How It Works</h2>
            <p style="font-size: 1.1em;">Using state-of-the-art deep learning models, this application analyzes the audio features of music, such as **tempo**, **pitch**, **spectral properties**, and **rhythmic patterns**, to classify songs into one of ten predefined genres. The process involves feature extraction with **Librosa**, data preprocessing, and prediction using a trained **TensorFlow model**.</p>

            <h2>Technology Stack</h2>
            <ul style="font-size: 1.1em;">
                <li><strong>TensorFlow:</strong> For building and training the deep learning model.</li>
                <li><strong>Librosa:</strong> For extracting audio features from music files.</li>
                <li><strong>Streamlit:</strong> For creating a simple, interactive, and user-friendly web interface.</li>
                <li><strong>Python:</strong> The primary programming language for implementing the project.</li>
            </ul>

            <h2>Features</h2>
            <ul style="font-size: 1.1em;">
                <li>Upload or Record audio files directly from your device.</li>
                <li>Get instant genre predictions with high accuracy.</li>
                <li>Easy-to-use interface for all users, regardless of technical expertise.</li>
            </ul>

            <h2>Future Enhancements</h2>
            <p style="font-size: 1.1em;">The current implementation classifies music into 10 genres, but we plan to expand it to support more genres and provide additional features like identifying instruments, moods, and BPM (beats per minute). Stay tuned!</p>

            <p style="text-align: center; font-size: 1.2em;">🎧 Let the music take over, one genre at a time! 🎶</p>
        """, unsafe_allow_html=True)

    elif app_mode == "Prediction":
        st.markdown("<h1 class='prediction-result'>Model Prediction <span class='rotating-icon'>🎼</span></h1>", unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
            <p style="font-size: 1.2em;">Choose whether to upload or record an audio file, and get the predicted genre along with confidence scores.</p>
        </div>
        """, unsafe_allow_html=True)

        # Option to upload or record audio
        input_type = "Upload Audio File" #Remove the radio button
        filepath = None

        if input_type == "Upload Audio File":
            test_audio = st.file_uploader("Upload an audio file", type=["mp3", "wav"], key="upload")
            if test_audio is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                    temp_file.write(test_audio.read())
                    filepath = temp_file.name
                st.audio(filepath)

        #Remove the "Record Audio" section
        # elif input_type == "Record Audio":
        #     col1, col2 = st.columns(2)
        #     with col1:
        #         duration = st.slider("Recording duration (seconds):", 1, 30, 15)
        #     with col2:
        #         if st.button("Start Recording"):
        #             filepath = record_audio(duration)
        #             if filepath:
        #                 st.audio(filepath)
        #                 st.session_state.recorded_file = filepath
        #                 st.success("Recording saved successfully!")

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
    Developed by Hassan Muhammad Yousuf | {st.session_state.username if st.session_state.logged_in else 'Please login'}
</div>
""", unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == "__main__":
    pass

