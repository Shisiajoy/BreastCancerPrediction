import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

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

# Streamlit app structure
st.title("Breast Cancer Prediction App")

# Explanation of how the app works
st.info("This app uses a quantized neural network model to predict the presence of breast cancer in mammogram images.")

# Expandable section for more information
with st.expander("Learn more about how the prediction is made"):
    st.markdown("""
    - **Model Details**: The model is a quantized neural network, optimized for inference on edge devices.
    - **Input Preprocessing**: The uploaded image is resized to 150x150 pixels and converted to grayscale.
    - **Prediction Threshold**: The model outputs a probability score between 0 and 1. A threshold of 0.5 is used to classify the result as either 'Cancer' or 'No Cancer'.
    - **Normalization**: Images are normalized to scale pixel values between 0 and 1 before being fed into the model.
    """)

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

        # Provide more context on the result
        if prediction_class == "Cancer":
            st.warning("The model detected signs of cancer. Please consult a healthcare professional for further evaluation.")
        else:
            st.success("The model did not detect signs of cancer. However, always consult with a healthcare professional for regular check-ups.")
