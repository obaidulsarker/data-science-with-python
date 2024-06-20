import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image


# Load the trained model
model_path = 'models/tuned_ghi_prediction_model.pkl'
model = joblib.load(model_path)

image_size = (128, 128) 

# Image preprocessing function
def preprocess_image(img_path):
    image = cv2.imread(img_path)
    image = cv2.resize(image, image_size)
    image = image / 255.0  # Normalize to [0, 1]
    
    # Add a batch dimension (1, height, width, channels)
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit app
def main():
    st.title("GHI Prediction from Sky Images")
    st.write("Upload a sky image to predict the Global Horizontal Irradiance (GHI).")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    work_dir='data/images'
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        file_name=uploaded_file.name
       
    
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        image_path =f'{work_dir}/{file_name}'
        st.write(image_path)
        
        # Preprocess the image
        preprocessed_image = preprocess_image(image_path)
        
        #images = np.array([preprocess_image(path) for path in data_filtered['image_path'].values])
        
        max_ghi = 600
        
        if preprocessed_image is not None:
            # Make a prediction
            with st.spinner('Predicting...'):
                prediction = model.predict(preprocessed_image)
                predicted_ghi = float(prediction[0, 0]) * max_ghi  # Extract the single value and convert to float
                st.success(f'Predicted GHI: {predicted_ghi:.2f} W/m^2')
        else:
            st.error("The image could not be processed for prediction.")


if __name__ == '__main__':
    main()