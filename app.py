import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the quantized model
try:
    interpreter = tf.lite.Interpreter(model_path='Model/quantized_model.tflite')
    interpreter.allocate_tensors()
except Exception as e:
    st.error(f"Error loading the quantized model: {e}")

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((150, 150))  # Adjust to match model's expected input size
    image = image.convert('L')  # Convert to grayscale
    img_array = np.array(image) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension (1 for grayscale)
    img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension
    return img_array

# Function to predict using the quantized model
def predict(image):
    # Preprocess the image
    input_data = preprocess_image(image)

    # Set input tensor
    try:
        interpreter.set_tensor(input_details[0]['index'], input_data)
    except Exception as e:
        st.error(f"Error setting input tensor: {e}")
        return None, None

    # Run inference
    try:
        interpreter.invoke()
    except Exception as e:
        st.error(f"Error during model inference: {e}")
        return None, None

    # Get the prediction
    try:
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction_probability = output_data[0][0]  # Assuming single output for binary classification
        return prediction_probability, output_data
    except Exception as e:
        st.error(f"Error retrieving prediction output: {e}")
        return None, None

# Streamlit app structure
st.title("Breast Cancer Prediction App")
st.write("This app uses a quantized neural network model to predict the presence of breast cancer in mammogram images.")

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
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict and display the result
        if st.button("Predict"):
            prediction_probability, raw_output = predict(image)

            if prediction_probability is not None:
                # Set a threshold to classify
                threshold = 0.5
                if prediction_probability > threshold:
                    st.write(f"**Prediction:** Cancer detected with a probability of {prediction_probability:.2f}")
                    st.warning("The model detected signs of cancer. Please consult a healthcare professional for further evaluation.")
                else:
                    st.write(f"**Prediction:** No cancer detected with a probability of {1 - prediction_probability:.2f}")
                    st.success("The model did not detect signs of cancer. However, always consult with a healthcare professional for regular check-ups.")

                # Debug: Print raw prediction
                st.write("Raw model output:", raw_output)

    except Exception as e:
        st.error(f"Error processing the image: {e}")
else:
    st.info("Please upload a mammogram image to get started.")
