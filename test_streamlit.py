import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image


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
        if st.button("Rotate Image", type="primary"):
            original_image = Image.open(img)
            rotated_image = original_image.rotate(180)
            st.image(rotated_image, caption='Rotated Image', width = 250)
    
###########################
## Importing Keras Model ##
###########################

st.write(f'tensorflow: {tf.__version__}')
st.write(f'streamlit: {st.__version__}')

from huggingface_hub import from_pretrained_keras 

try:
    model = from_pretrained_keras("abluna/dogbreed", token = "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")
    