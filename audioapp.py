from flask import Flask, jsonify, request
import tensorflow as tf
import numpy as np
import librosa

app = Flask(__name__)

class LivePredictions:
    def __init__(self, model_path):
        self.path = model_path
        self.interpreter = tf.lite.Interpreter(model_path=self.path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict_emotion(self, audio_file):
        try:
            data, sampling_rate = librosa.load(audio_file)
            mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
            x = np.expand_dims(mfccs, axis=0)
            x = np.expand_dims(x, axis=-1)
            self.interpreter.set_tensor(self.input_details[0]['index'], x.astype(np.float32))
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            predicted_class = np.argmax(output_data, axis=1)[0]
            emotion = self.convert_class_to_emotion(predicted_class)
            return emotion, None  # Returning emotion and no error
        except Exception as e:
            return None, str(e)  # Returning no emotion and error message

    @staticmethod
    def convert_class_to_emotion(pred):
        label_conversion = {
            0: 'neutral',
            1: 'calm',
            2: 'happy',
            3: 'sad',
            4: 'angry',
            5: 'fearful',
            6: 'disgust',
            7: 'surprised'
        }
        return label_conversion.get(pred, 'Unknown')

# Initialize the LivePredictions instance with your model path
model_path = 'emotion.tflite'
live_prediction = LivePredictions(model_path)
# Flask routes
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    audio_file = request.files['file']
    if audio_file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    
    try:
        emotion, error = live_prediction.predict_emotion(audio_file)
        if error:
            return jsonify({'error': f'Prediction failed: {error}'}), 500
        elif emotion:
            return jsonify({'emotion': emotion}), 200
        else:
            return jsonify({'error': 'Failed to predict emotion'}), 500
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')