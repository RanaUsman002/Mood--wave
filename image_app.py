import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from deepface import DeepFace

app = Flask(__name__)

# Emotion dictionary for mapping DeepFace emotions
emotion_dict = {
    'angry': 'anger', 
    'disgust': 'disgust', 
    'fear': 'fear', 
    'happy': 'happiness', 
    'sad': 'sadness', 
    'surprise': 'surprise', 
    'neutral': 'neutral'
}

@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image_np = np.array(image)

        # Analyze the image with DeepFace
        try:
            analysis = DeepFace.analyze(
                img_path=image_np,
                actions=['emotion'],
                enforce_detection=False  # Set enforce_detection to False
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        if not analysis or not isinstance(analysis, list) or len(analysis) == 0:
            return jsonify({"error": "No face detected"}), 400

        # Extract the dominant emotion
        dominant_emotion = analysis[0]['dominant_emotion']

        # Map DeepFace emotion to your emotion dictionary
        emotion = emotion_dict.get(dominant_emotion, "unknown")

        return jsonify({"emotions": emotion})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, port=56000, host='0.0.0.0')  # Listen on all interfaces
