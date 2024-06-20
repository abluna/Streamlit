import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

st.write("""
# Dog Classification Tool
Testing this app
""")

#########################
## Importing the image ##
#########################

img = st.file_uploader("Upload the image", type=None)

left_co,cent_co,last_co = st.columns(3)
with cent_co:   
    if img is not None:
        original_image = Image.open(img)
        st.image(original_image, caption='Your Image', width = 250)

#########################
## Exporting the image ##
#########################

left_co,cent_co,last_co = st.columns(3)
with cent_co:
    if img is not None:
        if st.button("Rotate Image"):
            original_image = Image.open(img)
            rotated_image = original_image.rotate(180)
            st.image(rotated_image, caption='Rotated Image', width = 250)
    
###########################
## Importing Keras Model ##
###########################

if img is not None:
    if st.button("Predict Breed"):
        with st.spinner('Wait for it...'):

            from huggingface_hub import from_pretrained_keras
            from tensorflow.keras.preprocessing import image
            
            model = from_pretrained_keras("abluna/dogbreed", token = "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")
               
             # `img` is a PIL image of size 224x224
            img_v2 = image.load_img(img, target_size=(250, 250))

            # `x` is a float32 Numpy array of shape (300, 300, 3)
            x = image.img_to_array(img_v2)

            # We add a dimension to transform our array into a "batch"
            # of size (1, 300, 300, 3)
            x = np.expand_dims(x, axis=0)

            # Finally we preprocess the batch
            # (this does channel-wise color normalization)
            x = preprocess_input(x)

            preds = model.predict(x)
            NumPredicted = np.argmax(preds[0])

            st.write(NumPredicted)
        
        