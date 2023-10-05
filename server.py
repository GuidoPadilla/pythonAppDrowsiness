from PIL import Image
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('final.h5')


@app.route('/predict', methods=['POST'])
def predict():
    # Ensure that the request contains an image file
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image_file = request.files['image']

    # Process the image using Pillow
    img = Image.open(image_file)
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    # Map class index to label
    class_labels = ["Open", "Closed"]
    predicted_label = class_labels[predicted_class]

    return jsonify({'prediction': predicted_label})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
