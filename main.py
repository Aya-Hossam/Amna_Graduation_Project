import flask
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

app = flask.Flask(__name__)
global model
model = load_model('survey.h5')  # Make sure to replace 'cnn_model(trained).h5' with your model path and format

image_path = None  # Define image_path globally

def some_function():
    global image_path
    image_path = "benign (7).png"  # Assign value to image_path within a function

# Call some_function to assign a value to image_path
some_function()

# Now you can use image_path anywhere in your code
print(image_path)  # Output: benign (7).png

def preprocess_image(image_path):
    # Example preprocessing function
    image = load_and_preprocess_image(image_path)
    image = np.expand_dims(image, axis=0)  # Add an extra dimension for batch size
    return image

def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((150, 150))
    image = image.convert('L')
    image = np.array(image)
    image = image.reshape((150, 150, 1))
    image = image.astype('float32') / 255.0
    return image

@app.route('/predict', methods=['POST'])
def predict():
    data = flask.request.get_json()  # Get input data from POST request
    image_path = data['image_path']  # Assuming you're passing the path to the image in the JSON data
    processed_image = preprocess_image(image_path)  # Preprocess the image
    prediction = model.predict(processed_image)  # Make prediction using the loaded model
    
    predicted_class_index = np.argmax(prediction[0])  # Get the predicted class index
    
    # Map the predicted class index back to its original label
    class_labels = ["benign", "malignant", "normal"]  
    predicted_class_label = class_labels[predicted_class_index]
    
    return flask.jsonify({'prediction': predicted_class_label})

if __name__ == '__main__':
    app.run(debug=True , port=7000)

#'cnn_model(trained).h5'