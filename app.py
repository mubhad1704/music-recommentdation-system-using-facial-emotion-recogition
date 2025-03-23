from flask import Flask, render_template, Response, request, jsonify
import cv2
import pygame
import os
import random
from deepface import DeepFace
from mtcnn import MTCNN
from time import time
import numpy as np
from statistics import mode
from collections import deque

app = Flask(__name__)

# Initialize pygame for music playback
pygame.mixer.init()

# Define emotions and their corresponding music folders with better mapping
EMOTION_MUSIC_PATHS = {
    "happy": "music/happy",
    "sad": "music/sad",
    "neutral": "music/neutral",
    "angry": "music/angry",     # Map angry to sad music
    "fear": "music/fear",  # Map fear to neutral music
    "surprise": "music/surprise",# Map surprise to happy music
    "disgust": "music/disgust"    # Map disgust to sad music
}

# Store the current song index and playlist
current_playlist = []
current_song_index = 0
current_song_start_time = 0
current_song_duration = 0
is_shuffle = False

# OpenCV Video Capture
video_capture = cv2.VideoCapture(0)
is_video_active = True

# Initialize MTCNN detector
detector = MTCNN()

# Emotion detection parameters
EMOTION_CONFIDENCE_THRESHOLD = 0.35  # Slightly lower threshold for better detection
EMOTION_DETECTION_FRAMES = 10  # Increased frame count for more stable detection
MIN_REQUIRED_FRAMES = 6  # Minimum number of successful detections needed
recent_emotions = deque(maxlen=EMOTION_DETECTION_FRAMES)

def preprocess_frame(frame):
    """Preprocess frame for better detection."""
    # Convert to RGB (DeepFace expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Enhance contrast using CLAHE
    lab = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl,a,b))
    
    # Convert back to RGB
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    # Apply slight Gaussian blur to reduce noise
    enhanced_rgb = cv2.GaussianBlur(enhanced_rgb, (3,3), 0)
    
    return enhanced_rgb

def align_face(frame, face_location):
    """Align face for better emotion detection."""
    try:
        x, y, w, h = face_location['box']
        # Add padding around the face
        padding_x = int(w * 0.1)
        padding_y = int(h * 0.1)
        
        # Ensure padded coordinates are within frame bounds
        start_x = max(0, x - padding_x)
        start_y = max(0, y - padding_y)
        end_x = min(frame.shape[1], x + w + padding_x)
        end_y = min(frame.shape[0], y + h + padding_y)
        
        face = frame[start_y:end_y, start_x:end_x]
        return cv2.resize(face, (224, 224))  # Standard size for most models
    except:
        return frame

def get_dominant_emotion(emotions_list):
    """Get the most confident and consistent emotion."""
    if not emotions_list:
        return None
    
    # Filter emotions with sufficient confidence
    confident_emotions = [e for e in emotions_list if max(e.values()) > EMOTION_CONFIDENCE_THRESHOLD]
    
    if not confident_emotions:
        return None
    
    # Calculate weighted votes for each emotion
    emotion_votes = {}
    for emotion_dict in confident_emotions:
        max_conf_emotion = max(emotion_dict.items(), key=lambda x: x[1])
        emotion_name = max_conf_emotion[0]
        confidence = max_conf_emotion[1]
        
        # Weight recent emotions more heavily
        weight = confident_emotions.index(emotion_dict) + 1
        emotion_votes[emotion_name] = emotion_votes.get(emotion_name, 0) + (confidence * weight)
    
    # Return the emotion with highest weighted votes
    if emotion_votes:
        return max(emotion_votes.items(), key=lambda x: x[1])[0]
    return None

def get_random_song(emotion):
    """Get a random song from the corresponding emotion folder."""
    global current_playlist, current_song_index, is_shuffle

    # Map similar emotions to main categories
    emotion_mapping = {
        "angry": "angry",
        "fear": "fear",
        "surprise": "surprise",
        "disgust": "disgust"
    }
    
    # Map emotion to main category if needed
    mapped_emotion = emotion_mapping.get(emotion, emotion)
    
    if mapped_emotion not in EMOTION_MUSIC_PATHS:
        return None

    folder = EMOTION_MUSIC_PATHS[mapped_emotion]
    songs = [os.path.join(folder, song) for song in os.listdir(folder) if song.endswith('.mp3')]

    if not songs:
        return None

    # Always shuffle the initial playlist
    current_playlist = songs.copy()  # Make a copy to preserve original order
    random.shuffle(current_playlist)
    current_song_index = 0
    
    # If shuffle is not enabled, sort the remaining songs after the first one
    if not is_shuffle and len(current_playlist) > 1:
        # Keep the first randomly selected song
        first_song = current_playlist[0]
        remaining_songs = current_playlist[1:]
        remaining_songs.sort()
        current_playlist = [first_song] + remaining_songs
    
    return current_playlist[0]

def play_music(song_path):
    """Play the selected music."""
    global current_song_start_time, current_song_duration
    pygame.mixer.music.stop()  # Stop current music
    pygame.mixer.music.load(song_path)
    sound = pygame.mixer.Sound(song_path)
    current_song_duration = sound.get_length()
    current_song_start_time = time()
    pygame.mixer.music.play()

@app.route('/')
def index():
    return render_template('index.html')

def generate_video():
    """Generate video frames for streaming."""
    global is_video_active
    while True:
        if not is_video_active:
            # Return a blank frame or stop the stream
            blank_frame = cv2.imread('static/blank.jpg') if os.path.exists('static/blank.jpg') else None
            if blank_frame is None:
                # Create a black frame if no blank image exists
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', blank_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            success, frame = video_capture.read()
            if not success:
                break
            else:
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Stream video feed to frontend."""
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    """Capture an image and detect emotion."""
    global current_playlist, current_song_index, is_video_active, recent_emotions
    
    # Activate video feed
    is_video_active = True
    recent_emotions.clear()  # Clear previous emotions
    successful_detections = 0
    
    # Collect multiple frames for more stable detection
    for _ in range(EMOTION_DETECTION_FRAMES):
        success, frame = video_capture.read()
        if not success:
            continue

        try:
            # Preprocess frame
            processed_frame = preprocess_frame(frame)
            
            # Detect face
            faces = detector.detect_faces(processed_frame)
            if not faces:
                continue

            # Get the face with highest confidence
            face = max(faces, key=lambda x: x['confidence'])
            if face['confidence'] < 0.95:  # Increased confidence threshold for face detection
                continue

            # Align face
            aligned_face = align_face(processed_frame, face)
            
            # Analyze emotion
            emotion_analysis = DeepFace.analyze(aligned_face, actions=['emotion'], enforce_detection=False)
            if emotion_analysis:
                recent_emotions.append(emotion_analysis[0]['emotion'])
                successful_detections += 1
        
        except Exception as e:
            continue

    # If we couldn't get enough successful detections
    if successful_detections < MIN_REQUIRED_FRAMES:
        return jsonify({
            "error": f"Could not detect face consistently. Please ensure good lighting and face the camera directly. (Got {successful_detections}/{MIN_REQUIRED_FRAMES} successful detections)"
        })

    # Get the dominant emotion
    dominant_emotion = get_dominant_emotion(recent_emotions)
    if not dominant_emotion:
        return jsonify({
            "error": "Could not determine emotion confidently. Please try again with better lighting."
        })

    # Get a song based on detected emotion
    song = get_random_song(dominant_emotion)
    if song:
        play_music(song)
        # Deactivate video feed after successful detection
        is_video_active = False

    return jsonify({
        "emotion": dominant_emotion,
        "song": song,
        "video_active": is_video_active,
        "confidence": max(recent_emotions[-1].values()),  # Return confidence of last detection
        "successful_frames": successful_detections
    })

@app.route('/toggle_video', methods=['POST'])
def toggle_video():
    """Toggle the video feed state."""
    global is_video_active
    is_video_active = request.json.get('active', True)
    return jsonify({"video_active": is_video_active})

@app.route('/play')
def play():
    """Play the current song."""
    pygame.mixer.music.unpause()
    return jsonify({"status": "playing"})

@app.route('/pause')
def pause():
    """Pause the current song."""
    pygame.mixer.music.pause()
    return jsonify({"status": "paused"})

@app.route('/stop')
def stop():
    """Stop music playback."""
    pygame.mixer.music.stop()
    return jsonify({"status": "stopped"})

@app.route('/toggle_shuffle', methods=['POST'])
def toggle_shuffle():
    """Toggle shuffle mode and reorganize playlist if needed."""
    global is_shuffle, current_playlist, current_song_index
    
    is_shuffle = request.json.get('shuffle', False)
    
    if current_playlist:
        # Save current song
        current_song = current_playlist[current_song_index]
        
        if is_shuffle:
            # Shuffle playlist but keep current song at current index
            remaining_songs = current_playlist[:current_song_index] + current_playlist[current_song_index + 1:]
            random.shuffle(remaining_songs)
            current_playlist = remaining_songs[:current_song_index] + [current_song] + remaining_songs[current_song_index:]
        else:
            # Restore original order based on filenames
            current_playlist.sort()
            # Find new index of current song
            current_song_index = current_playlist.index(current_song)
    
    return jsonify({
        "shuffle": is_shuffle,
        "current_index": current_song_index,
        "playlist_length": len(current_playlist) if current_playlist else 0
    })

@app.route('/next')
def next_song():
    """Play the next song in the playlist."""
    global current_song_index, current_playlist, is_shuffle

    if not current_playlist:
        return jsonify({"error": "No playlist available"})

    if current_song_index < len(current_playlist) - 1:
        current_song_index += 1
    else:
        # If at the end of playlist and shuffle is on, reshuffle the playlist
        if is_shuffle:
            random.shuffle(current_playlist)
        current_song_index = 0

    play_music(current_playlist[current_song_index])
    return jsonify({"status": "playing", "song": current_playlist[current_song_index]})

@app.route('/previous')
def previous_song():
    """Play the previous song in the playlist."""
    global current_song_index, current_playlist

    if not current_playlist:
        return jsonify({"error": "No playlist available"})

    if current_song_index > 0:
        current_song_index -= 1
    else:
        current_song_index = len(current_playlist) - 1

    play_music(current_playlist[current_song_index])
    return jsonify({"status": "playing", "song": current_playlist[current_song_index]})

@app.route('/set_volume', methods=['POST'])
def set_volume():
    """Set the volume level."""
    try:
        volume = float(request.json.get('volume', 1.0))
        # Ensure volume is between 0 and 1
        volume = max(0.0, min(1.0, volume))
        pygame.mixer.music.set_volume(volume)
        return jsonify({"status": "success", "volume": volume})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/seek', methods=['POST'])
def seek():
    """Seek to a specific position in the song."""
    try:
        position = float(request.json.get('position', 0))
        # Convert position from seconds to milliseconds
        pygame.mixer.music.set_pos(position)
        global current_song_start_time
        current_song_start_time = time() - position
        return jsonify({"status": "success", "position": position})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/get_song_position')
def get_song_position():
    """Get the current position and duration of the playing song."""
    global current_song_start_time, current_song_duration
    
    if not pygame.mixer.music.get_busy():
        return jsonify({
            "position": 0,
            "duration": current_song_duration,
            "volume": pygame.mixer.music.get_volume()
        })
    
    current_position = time() - current_song_start_time
    if current_position > current_song_duration:
        current_position = current_song_duration
    
    return jsonify({
        "position": current_position,
        "duration": current_song_duration,
        "volume": pygame.mixer.music.get_volume()
    })

if __name__ == '__main__':
    app.run(debug=True)
