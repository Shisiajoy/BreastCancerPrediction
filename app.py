import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the quantized model
interpreter = tf.lite.Interpreter(model_path='Model/quantized_model.tflite')
interpreter.allocate_tensors()

def predict(image):
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    # Preprocess image
    image = image.resize((150, 150)).convert('RGB')
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # Run inference
    interpreter.set_tensor(input_details['index'], image_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details['index'])[0][0]
    
    # Determine the result
    probability = prediction
    label = "Cancer" if probability >= 0.5 else "No Cancer"
    return label, probability

st.title("Breast Cancer Prediction App")
st.write("Upload a Mammogram Image")

uploaded_file = st.file_uploader("Drag and drop file here", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    label, probability = predict(image)
    st.write(f"Prediction: {label} with a probability of {probability:.2f}")

    # Add more detailed information
    st.write("### Confidence Score")
    st.write(f"The confidence score of the prediction is: {probability:.2f}")


# Add a sidebar with additional information or instructions
st.sidebar.header("Instructions")
st.sidebar.write("1. Upload a mammogram image.")
st.sidebar.write("2. The model will predict whether it shows signs of cancer or not.")
st.sidebar.write("3. The result will include a probability score.")


import matplotlib.pyplot as plt

def plot_prediction_distribution(probability):
    fig, ax = plt.subplots()
    ax.bar(['Cancer', 'No Cancer'], [probability, 1 - probability])
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Distribution')
    st.pyplot(fig)

# After getting the prediction
plot_prediction_distribution(probability)

