from flask import Flask, request, jsonify
from flask import send_from_directory
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = load_model('image_classification_model.h5')

@app.route('/')
def index():
    return "Welcome to the Image Classification API!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    img = image.load_img(file, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    prediction = model.predict(img_array)
    result = 'Cat' if prediction[0] > 0.5 else 'Dog'
    
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
