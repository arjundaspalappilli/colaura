from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Path to the folder where uploaded images are saved
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model
model = tf.keras.models.load_model('cnn_model.h5')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    """Check if the uploaded file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess the image without resizing."""
    img = Image.open(image_path)
    img_array = np.array(img) / 255.0  # Normalize the image to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Route for the home page (for uploading images)
@app.route('/', methods=['GET'])
def home():
    """Render the home page for file upload."""
    return render_template('home.html')

# Route for handling the image upload and prediction
@app.route('/result', methods=['POST'])
def result():
    """Handle file upload and prediction."""
    if 'file' not in request.files:
        return redirect(url_for('home'))
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)  # Secure the filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)  # Save the uploaded file

        # Preprocess the uploaded image (without resizing)
        img_array = preprocess_image(filepath)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_label = np.round(prediction).astype(int)[0][0]  # Get the predicted label

        # Determine the label based on the prediction
        label = 'REAL' if predicted_label == 1 else 'FAKE'

        # Render the result page with the prediction and image
        return render_template('result.html', label=label, image_path=filepath)

    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)
