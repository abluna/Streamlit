import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions

st.write("""
# Dog Classification Tool
The goal of this tool is to quickly predict a dog's breed based on a single image. \n
To test a breed prediction, upload a photo and click 'Predict Breed'
""")


######################
## Import the model ##
######################

@st.experimental_singleton
def load_model():
    from huggingface_hub import from_pretrained_keras
    model = from_pretrained_keras("abluna/dogbreed", token = "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")
    return model

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

#
#left_co,cent_co,last_co = st.columns(3)
#with cent_co:
#    if img is not None:
#        if st.button("Rotate Image"):
#            original_image = Image.open(img)
#            rotated_image = original_image.rotate(180)
#            st.image(rotated_image, caption='Rotated Image', width = 250)
 
###########################
## Importing Keras Model ##
###########################

index_list = {'Affenpinscher': 0,
 'Afghan hound': 1,
 'Airedale terrier': 2,
 'Akita': 3,
 'Alaskan malamute': 4,
 'American eskimo dog': 5,
 'American foxhound': 6,
 'American staffordshire terrier': 7,
 'American water spaniel': 8,
 'Anatolian shepherd dog': 9,
 'Australian cattle dog': 10,
 'Australian shepherd': 11,
 'Australian terrier': 12,
 'Basenji': 13,
 'Basset hound': 14,
 'Beagle': 15,
 'Bearded collie': 16,
 'Beauceron': 17,
 'Bedlington terrier': 18,
 'Belgian malinois': 19,
 'Belgian sheepdog': 20,
 'Belgian tervuren': 21,
 'Bernese mountain dog': 22,
 'Bichon frise': 23,
 'Black and tan coonhound': 24,
 'Black russian terrier': 25,
 'Bloodhound': 26,
 'Bluetick coonhound': 27,
 'Border collie': 28,
 'Border terrier': 29,
 'Borzoi': 30,
 'Boston terrier': 31,
 'Bouvier des flandres': 32,
 'Boxer': 33,
 'Boykin spaniel': 34,
 'Briard': 35,
 'Brittany': 36,
 'Brussels griffon': 37,
 'Bull terrier': 38,
 'Bulldog': 39,
 'Bullmastiff': 40,
 'Cairn terrier': 41,
 'Canaan dog': 42,
 'Cane corso': 43,
 'Cardigan welsh corgi': 44,
 'Cavalier king charles spaniel': 45,
 'Chesapeake bay retriever': 46,
 'Chihuahua': 47,
 'Chinese crested': 48,
 'Chinese shar-pei': 49,
 'Chow chow': 50,
 'Clumber spaniel': 51,
 'Cocker spaniel': 52,
 'Collie': 53,
 'Curly-coated retriever': 54,
 'Dachshund': 55,
 'Dalmatian': 56,
 'Dandie dinmont terrier': 57,
 'Doberman pinscher': 58,
 'Dogue de bordeaux': 59,
 'English cocker spaniel': 60,
 'English setter': 61,
 'English springer spaniel': 62,
 'English toy spaniel': 63,
 'Entlebucher mountain dog': 64,
 'Field spaniel': 65,
 'Finnish spitz': 66,
 'Flat-coated retriever': 67,
 'French bulldog': 68,
 'German pinscher': 69,
 'German shepherd dog': 70,
 'German shorthaired pointer': 71,
 'German wirehaired pointer': 72,
 'Giant schnauzer': 73,
 'Glen of imaal terrier': 74,
 'Golden retriever': 75,
 'Gordon setter': 76,
 'Great dane': 77,
 'Great pyrenees': 78,
 'Greater swiss mountain dog': 79,
 'Greyhound': 80,
 'Havanese': 81,
 'Ibizan hound': 82,
 'Icelandic sheepdog': 83,
 'Irish red and white setter': 84,
 'Irish setter': 85,
 'Irish terrier': 86,
 'Irish water spaniel': 87,
 'Irish wolfhound': 88,
 'Italian greyhound': 89,
 'Japanese chin': 90,
 'Keeshond': 91,
 'Kerry blue terrier': 92,
 'Komondor': 93,
 'Kuvasz': 94,
 'Labrador retriever': 95,
 'Lakeland terrier': 96,
 'Leonberger': 97,
 'Lhasa apso': 98,
 'Lowchen': 99,
 'Maltese': 100,
 'Manchester terrier': 101,
 'Mastiff': 102,
 'Miniature schnauzer': 103,
 'Neapolitan mastiff': 104,
 'Newfoundland': 105,
 'Norfolk terrier': 106,
 'Norwegian buhund': 107,
 'Norwegian elkhound': 108,
 'Norwegian lundehund': 109,
 'Norwich terrier': 110,
 'Nova scotia duck tolling retriever': 111,
 'Old english sheepdog': 112,
 'Otterhound': 113,
 'Papillon': 114,
 'Parson russell terrier': 115,
 'Pekingese': 116,
 'Pembroke welsh corgi': 117,
 'Petit basset griffon vendeen': 118,
 'Pharaoh hound': 119,
 'Plott': 120,
 'Pointer': 121,
 'Pomeranian': 122,
 'Poodle': 123,
 'Portuguese water dog': 124,
 'Saint bernard': 125,
 'Silky terrier': 126,
 'Smooth fox terrier': 127,
 'Tibetan mastiff': 128,
 'Welsh springer spaniel': 129,
 'Wirehaired pointing griffon': 130,
 'Xoloitzcuintli': 131,
 'Yorkshire terrier': 132}

if img is not None:
    if st.button("Predict Breed"):
        with st.spinner('Wait for it...'):
        
            # Use the function to load your data
            tf_model = load_model()
                         
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

            preds = tf_model.predict(x)
            
            ## Get list of predictions
            pred_dict = dict(zip(index_list, np.round(preds[0]*100,2)))
            Sorted_Prediction_Dictionary = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)

            Count_5Perc = preds[0][preds[0]>0.02]

            if len(Count_5Perc) == 1:
                TopPredictions = Sorted_Prediction_Dictionary[0]
                to_df = list(TopPredictions)
                df = pd.DataFrame({'Breed': to_df[0], 'Probability':to_df[1]}, index=[0]) 
            if len(Count_5Perc) > 1:
                TopPredictions = Sorted_Prediction_Dictionary[0:len(Count_5Perc)]
                df = pd.DataFrame(TopPredictions, columns =['Breed', 'Probability'])
            
            df['Probability'] = df['Probability'].round(1)
            
            st.table(df)
            