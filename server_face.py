from PIL import Image
from flask import Flask, request, jsonify
import tensorflow as tf
import dlib
import cv2
import numpy as np

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('final_alternativo.h5')

# Load the Dlib face and eye detectors
face_detector = dlib.get_frontal_face_detector()
eye_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


@app.route('/predict', methods=['POST'])
def predict():
    # Ensure that the request contains an image file
    if 'image' not in request.files:
        # Return 400 Bad Request
        return jsonify({'error': 'No image provided'}), 400

    try:
        image_file = request.files['image']

        # Process the image using Pillow
        img = Image.open(image_file)
        img = np.array(img)

        # Perform face detection using Dlib
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('gray_image.png', gray)
        faces = face_detector(gray)

        # Initialize eyes_roi
        eyes_roi = None

        for face in faces:
            # Get face landmarks
            landmarks = eye_detector(gray, face)

            # Extract and resize left eye
            # Adjust these values as needed
            left_eye_x = max(0, landmarks.part(36).x -
                             35)  # Extend by 35 pixels
            left_eye_y = max(0, landmarks.part(36).y -
                             35)  # Extend by 35 pixels
            left_eye_width = landmarks.part(
                39).x - landmarks.part(36).x + 70  # Extend by 70 pixels
            left_eye_height = landmarks.part(
                39).y - landmarks.part(36).y + 70  # Extend by 70 pixels

            left_eye = img[left_eye_y:left_eye_y + left_eye_height,
                           left_eye_x:left_eye_x + left_eye_width]
            left_eye = cv2.resize(left_eye, (224, 224))
            left_eye = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)

            # Extract and resize right eye with an extended region
            right_eye_x = max(0, landmarks.part(
                42).x - 35)  # Extend by 35 pixels
            right_eye_y = max(0, landmarks.part(
                42).y - 35)  # Extend by 35 pixels
            right_eye_width = landmarks.part(
                45).x - landmarks.part(42).x + 70  # Extend by 70 pixels
            right_eye_height = landmarks.part(
                45).y - landmarks.part(42).y + 70  # Extend by 70 pixels

            right_eye = img[right_eye_y:right_eye_y + right_eye_height,
                            right_eye_x:right_eye_x + right_eye_width]
            right_eye = cv2.resize(right_eye, (224, 224))
            right_eye = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)

            # Combine both eyes
            # Duplicate the grayscale channel
            eyes_roi = np.dstack((left_eye, right_eye, right_eye))
            cv2.imwrite('eyes_of_gray.png', eyes_roi)

            break  # Break after detecting eyes in the first face

        if eyes_roi is not None:
            img_array = np.expand_dims(eyes_roi, axis=0)
            img_array = img_array.astype('float32') / 255.0

            # Make predictions
            predictions = model.predict(img_array)

            if predictions[0][0] >= 0.5:
                return jsonify({'prediction': "Closed"}), 200  # Return 200 OK
            else:
                return jsonify({'prediction': "Open"}), 200  # Return 200 OK
        else:
            # Return 400 Bad Request
            return jsonify({"error": 'Eyes not detected'}), 400

    except Exception as e:
        # Return 400 Bad Request
        return jsonify({"error": 'UNEXPECTED ERROR FROM CLASSIFIER' + str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
