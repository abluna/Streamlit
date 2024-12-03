import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions

st.write("""
# Dog or Bird Classification Tool
The goal of this tool is to quickly predict a dog's breed or bird species based on a single image. \n
To test a breed prediction, upload a photo and click 'Predict'
""")

option = st.selectbox(
    "I want to predict a...",
    ("Bird", "Dog"),
    index=None,
    placeholder="Select Animal",
)

st.write("You selected:", option)

######################
## Import the model ##
######################

@st.cache_resource
def load_model():
    model = keras.saving.load_model('hf://abluna/dogbreedv20')
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
 
###########################
## Importing Keras Model ##
###########################

index_list = {'AFFENPINSCHER': 0,
 'AIREDALE TERRIER': 1,
 'AIREDALE TERRIER CROSS': 2,
 'AKITA': 3,
 'AKITA CROSS': 4,
 'ALASKAN MALAMUTE': 5,
 'ALASKAN MALAMUTE CROSS': 6,
 'AM STAFF': 7,
 'AM STAFF CROSS': 8,
 'AMERICAN BULLDOG': 9,
 'AMERICAN BULLDOG CROSS': 10,
 'AMERICAN STAFFORDSHIRE BULL TERRIER': 11,
 'AMERICAN STAFFORDSHIRE BULL TERRIER CROSS': 12,
 'AMSTAFF': 13,
 'AMSTAFF CROSS': 14,
 'ANATOLIAN SHEPHERD DOG': 15,
 'ANATOLIAN SHEPHERD DOG CROSS': 16,
 'AUSTRALIAN BULLDOG': 17,
 'AUSTRALIAN BULLDOG CROSS': 18,
 'AUSTRALIAN CATTLE DOG': 19,
 'AUSTRALIAN CATTLE DOG CROSS': 20,
 'AUSTRALIAN SHEPHERD': 21,
 'AUSTRALIAN SHEPHERD CROSS': 22,
 'AUSTRALIAN TERRIER': 23,
 'AUSTRALIAN TERRIER CROSS': 24,
 'BANDOG': 25,
 'BANDOG CROSS': 26,
 'BASENJI': 27,
 'BASENJI CROSS': 28,
 'BASSET HOUND CROSS': 29,
 'BEAGLE': 30,
 'BEAGLE CROSS': 31,
 'BEARDED COLLIE': 32,
 'BEARDED COLLIE CROSS': 33,
 'BELGIAN MALINOIS': 34,
 'BELGIAN MALINOIS CROSS': 35,
 'BELGIAN SHEPHERD': 36,
 'BELGIAN SHEPHERD CROSS': 37,
 'BELGIAN TERVUREN': 38,
 'BICHON FRISÉ': 39,
 'BICHON FRISÉ CROSS': 40,
 'BLACK MOUTH CUR CROSS': 41,
 'BLOODHOUND': 42,
 'BLOODHOUND CROSS': 43,
 'BOERBOEL': 44,
 'BOERBOEL CROSS': 45,
 'BORDER COLLIE': 46,
 'BORDER COLLIE CROSS': 47,
 'BORDER TERRIER': 48,
 'BORDER TERRIER CROSS': 49,
 'BORZOI': 50,
 'BORZOI CROSS': 51,
 'BOSTON TERRIER': 52,
 'BOSTON TERRIER CROSS': 53,
 'BOXER': 54,
 'BOXER CROSS': 55,
 'BRITISH BULLDOG': 56,
 'BRITISH BULLDOG CROSS': 57,
 'BRITTANY': 58,
 'BRUSSELS GRIFFON CROSS': 59,
 'BULL ARAB': 60,
 'BULL ARAB CROSS': 61,
 'BULL TERRIER': 62,
 'BULL TERRIER CROSS': 63,
 'BULLDOG': 64,
 'BULLDOG CROSS': 65,
 'BULLMASTIFF': 66,
 'BULLMASTIFF CROSS': 67,
 'CAIRN TERRIER': 68,
 'CAIRN TERRIER CROSS': 69,
 'CANE CORSO': 70,
 'CANE CORSO CROSS': 71,
 'CATAHOULA': 72,
 'CATAHOULA CROSS': 73,
 'CAVALIER KING CHARLES': 74,
 'CAVALIER KING CHARLES CROSS': 75,
 'CHESAPEAKE BAY RETRIEVER': 76,
 'CHIHUAHUA': 77,
 'CHIHUAHUA CROSS': 78,
 'CHINESE CRESTED': 79,
 'CHINESE CRESTED CROSS': 80,
 'CHINESE SHAR-PEI': 81,
 'CHINESE SHAR-PEI CROSS': 82,
 'CHOW CHOW': 83,
 'COCKER SPANIEL': 84,
 'COCKER SPANIEL CROSS': 85,
 'COLLIE CROSS': 86,
 'COONHOUND': 87,
 'COONHOUND CROSS': 88,
 'CORGI': 89,
 'CORGI CROSS': 90,
 'CROSS BREED': 91,
 'CROSS BREED CROSS': 92,
 'CURLY-COATED RETRIEVER': 93,
 'CURLY-COATED RETRIEVER CROSS': 94,
 'DACHSHUND': 95,
 'DACHSHUND CROSS': 96,
 'DALMATIAN': 97,
 'DALMATIAN CROSS': 98,
 'DEER HOUND CROSS': 99,
 'DINGO': 100,
 'DINGO CROSS': 101,
 'DOBERMAN PINSCHER': 102,
 'DOBERMAN PINSCHER CROSS': 103,
 'DOGUE DE BORDEAUX': 104,
 'DOGUE DE BORDEAUX CROSS': 105,
 'DUTCH SHEPHERD CROSS': 106,
 'ELKHOUND': 107,
 'ENGLISH BULLDOG': 108,
 'ENGLISH BULLDOG CROSS': 109,
 'ENGLISH FOXHOUND': 110,
 'ENGLISH FOXHOUND CROSS': 111,
 'ENGLISH POINTER': 112,
 'ENGLISH POINTER CROSS': 113,
 'ENGLISH STAFFY': 114,
 'ENGLISH STAFFY CROSS': 115,
 'FOX TERRIER': 116,
 'FOX TERRIER CROSS': 117,
 'FOXHOUND': 118,
 'FOXHOUND CROSS': 119,
 'FRENCH BULLDOG': 120,
 'FRENCH BULLDOG CROSS': 121,
 'FRENCH MASTIFF': 122,
 'FRENCH MASTIFF CROSS': 123,
 'GERMAN PINSCHER': 124,
 'GERMAN PINSCHER CROSS': 125,
 'GERMAN SHEPHERD': 126,
 'GERMAN SHEPHERD CROSS': 127,
 'GERMAN SHORTHAIRED POINTER': 128,
 'GERMAN SHORTHAIRED POINTER CROSS': 129,
 'GERMAN SPITZ': 130,
 'GERMAN SPITZ CROSS': 131,
 'GERMAN WIREHAIRED POINTER': 132,
 'GERMAN WIREHAIRED POINTER CROSS': 133,
 'GOLDEN RETRIEVER': 134,
 'GOLDEN RETRIEVER CROSS': 135,
 'GOLDENDOODLE': 136,
 'GOLDENDOODLE CROSS': 137,
 'GORDON SETTER': 138,
 'GREAT DANE': 139,
 'GREAT DANE CROSS': 140,
 'GREATER SWISS MOUNTAIN DOG CROSS': 141,
 'GREYHOUND': 142,
 'GREYHOUND CROSS': 143,
 'GROENDALE SHEPHERD CROSS': 144,
 'HARRIER': 145,
 'HARRIER CROSS': 146,
 'HEINZ 57': 147,
 'HEINZ 57 CROSS': 148,
 'HUNTAWAY': 149,
 'HUNTAWAY CROSS': 150,
 'IBIZAN HOUND': 151,
 'IRISH SETTER': 152,
 'IRISH TERRIER': 153,
 'ITALIAN GREYHOUND': 154,
 'ITALIAN GREYHOUND CROSS': 155,
 'ITALIAN MASTIFF CROSS': 156,
 'JACK RUSSELL TERRIER': 157,
 'JACK RUSSELL TERRIER CROSS': 158,
 'JAPANESE SPITZ': 159,
 'JAPANESE SPITZ CROSS': 160,
 'KEESHOND': 161,
 'KELPIE': 162,
 'KELPIE CROSS': 163,
 'KOOLIE': 164,
 'KOOLIE CROSS': 165,
 'LABRADOODLE': 166,
 'LABRADOODLE CROSS': 167,
 'LABRADOR RETRIEVER': 168,
 'LABRADOR RETRIEVER CROSS': 169,
 'LEONBERGER CROSS': 170,
 'LHASA APSO': 171,
 'LHASA APSO CROSS': 172,
 'MALTESE': 173,
 'MALTESE CROSS': 174,
 'MAREMMA SHEEPDOG': 175,
 'MAREMMA SHEEPDOG CROSS': 176,
 'MASTIFF': 177,
 'MASTIFF CROSS': 178,
 'MINI FOX TERRIER': 179,
 'MINI FOX TERRIER CROSS': 180,
 'MINIATURE BULL TERRIER': 181,
 'MINIATURE BULL TERRIER CROSS': 182,
 'MINIATURE PINSCHER': 183,
 'MINIATURE PINSCHER CROSS': 184,
 'MINIATURE POODLE': 185,
 'MINIATURE POODLE CROSS': 186,
 'MINIATURE SCHNAUZER': 187,
 'MINIATURE SCHNAUZER CROSS': 188,
 'MIXED BREED': 189,
 'MIXED BREED CROSS': 190,
 'MURRAY VALLEY CC RETREIVER CROSS': 191,
 'NEOPOLITAN MASTIFF': 192,
 'NEOPOLITAN MASTIFF CROSS': 193,
 'NEWFOUNDLAND': 194,
 'OLD ENGLISH SHEEPDOG CROSS': 195,
 'PAPILLON': 196,
 'PAPILLON CROSS': 197,
 'PEKINGESE': 198,
 'PHARAOH HOUND CROSS': 199,
 'PIT BULL TERRIER': 200,
 'PIT BULL TERRIER CROSS': 201,
 'POINTER': 202,
 'POINTER CROSS': 203,
 'POMERANIAN': 204,
 'POMERANIAN CROSS': 205,
 'POODLE': 206,
 'POODLE - STANDARD': 207,
 'POODLE - STANDARD CROSS': 208,
 'POODLE - TOY': 209,
 'POODLE - TOY CROSS': 210,
 'POODLE CROSS': 211,
 'PORTUGUESE WATER DOG': 212,
 'PUG': 213,
 'PUG CROSS': 214,
 'PULI CROSS': 215,
 'RHODESIAN RIDGEBACK': 216,
 'RHODESIAN RIDGEBACK CROSS': 217,
 'ROTTWEILER': 218,
 'ROTTWEILER CROSS': 219,
 'ROUGH COLLIE': 220,
 'ROUGH COLLIE CROSS': 221,
 'SAINT BERNARD': 222,
 'SAINT BERNARD CROSS': 223,
 'SAMOYED': 224,
 'SAMOYED CROSS': 225,
 'SARPLANINAC': 226,
 'SCOTCH COLLIE': 227,
 'SCOTTISH TERRIER CROSS': 228,
 'SHELTIE': 229,
 'SHETLAND SHEEPDOG': 230,
 'SHETLAND SHEEPDOG CROSS': 231,
 'SHIH TZU': 232,
 'SHIH TZU CROSS': 233,
 'SIBERIAN HUSKY': 234,
 'SIBERIAN HUSKY CROSS': 235,
 'SILKY TERRIER': 236,
 'SILKY TERRIER CROSS': 237,
 'SMITHFIELD': 238,
 'SMITHFIELD CROSS': 239,
 'SPITZ': 240,
 'SPITZ CROSS': 241,
 'SPRINGER SPANIEL': 242,
 'SPRINGER SPANIEL CROSS': 243,
 'STAFFORDSHIRE BULL TERRIER': 244,
 'STAFFORDSHIRE BULL TERRIER CROSS': 245,
 'STAFFY': 246,
 'STAFFY CROSS': 247,
 'STAGHOUND': 248,
 'STAGHOUND CROSS': 249,
 'STANDARD SCHNAUZER': 250,
 'STANDARD SCHNAUZER CROSS': 251,
 'TERRIER': 252,
 'TERRIER CROSS': 253,
 'TERRIER CROSS CROSS': 254,
 'TIBETAN MASTIFF': 255,
 'TIBETAN MASTIFF CROSS': 256,
 'TIBETAN SPANIEL': 257,
 'TIBETAN SPANIEL CROSS': 258,
 'VIZSLA': 259,
 'VIZSLA CROSS': 260,
 'WATER SPANIEL CROSS': 261,
 'WEIMARANER': 262,
 'WEIMARANER CROSS': 263,
 'WEST HIGHLAND WHITE TERRIER': 264,
 'WEST HIGHLAND WHITE TERRIER CROSS': 265,
 'WHEATEN TERRIER': 266,
 'WHIPPET': 267,
 'WHIPPET CROSS': 268,
 'WHITE SWISS SHEPHERD': 269,
 'WHITE SWISS SHEPHERD CROSS': 270,
 'WOLFHOUND': 271,
 'WOLFHOUND CROSS': 272,
 'YORKSHIRE TERRIER': 273,
 'YORKSHIRE TERRIER CROSS': 274}

if img is not None:
    if st.button("Predict Breed"):
        with st.spinner('Wait for it...'):
        
            # Use the function to load your data
            tf_model = load_model()
                         
             # `img` is a PIL image of size 224x224
            img_v2 = image.load_img(img, target_size=(350, 350))

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
            
            df['Probability'] = df['Probability'].round(2)

            st.dataframe(
            df,
            column_config={
                "name": "App name",
                "Probability": st.column_config.NumberColumn(
                    "Probability",
                    format='%.2f %%'
                )
            },
            hide_index=True,
        )



            