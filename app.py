import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the quantized model
interpreter = tf.lite.Interpreter(model_path='Model/quantized_model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = image.resize((150, 150))  # Resize the image to 150x150
    img = img.convert('L')  # Convert to grayscale
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension (1 for grayscale)
    img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension
    img_array = img_array.astype('float32') / 255.0  # Normalize the image
    return img_array

# Function to predict using the quantized model
def predict(image):
    # Preprocess the image
    input_data = preprocess_image(image)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the prediction
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction_probability = output_data[0][0]

    # Classify based on the threshold
    threshold = 0.5
    if prediction_probability >= threshold:
        prediction_class = "Cancer"
    else:
        prediction_class = "No Cancer"

    return prediction_class, prediction_probability


def display_layer_activations(image, interpreter, input_details, output_details):
    # Preprocess the image
    input_data = preprocess_image(image)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Simulate displaying convolution layers
    # Example: Let's assume we have 3 convolution layers to visualize
    layer_activations = [input_data[0, :, :, 0]]  # Start with the original grayscale image

    # Manually simulate activations (this part will differ based on your actual layers)
    for i in range(3):
        # Simulate a convolution layer by adding a random pattern to the image (for demonstration)
        layer_activations.append(np.random.rand(150, 150))  # Placeholder for actual layer outputs

    # Display the activations using Streamlit
    st.write("### Convolution Layer Activations")
    for i, activation in enumerate(layer_activations):
        plt.imshow(activation, cmap='gray')
        st.image(plt, caption=f"Layer {i+1} Activation", use_column_width=True)


# Streamlit app structure
st.title("Breast Cancer Prediction App")

# Upload image section
uploaded_file = st.file_uploader("Upload a Mammogram Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict and display the result
    if st.button("Predict"):
        prediction_class, prediction_probability = predict(image)
        st.write(f"**Prediction:** {prediction_class} with a probability of {prediction_probability:.2f}")
        
        # Visualize intermediate layers
        display_layer_activations(image, interpreter, input_details, output_details)
