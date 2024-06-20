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

index_list = {'001.Affenpinscher': 0,
 '002.Afghan_hound': 1,
 '003.Airedale_terrier': 2,
 '004.Akita': 3,
 '005.Alaskan_malamute': 4,
 '006.American_eskimo_dog': 5,
 '007.American_foxhound': 6,
 '008.American_staffordshire_terrier': 7,
 '009.American_water_spaniel': 8,
 '010.Anatolian_shepherd_dog': 9,
 '011.Australian_cattle_dog': 10,
 '012.Australian_shepherd': 11,
 '013.Australian_terrier': 12,
 '014.Basenji': 13,
 '015.Basset_hound': 14,
 '016.Beagle': 15,
 '017.Bearded_collie': 16,
 '018.Beauceron': 17,
 '019.Bedlington_terrier': 18,
 '020.Belgian_malinois': 19,
 '021.Belgian_sheepdog': 20,
 '022.Belgian_tervuren': 21,
 '023.Bernese_mountain_dog': 22,
 '024.Bichon_frise': 23,
 '025.Black_and_tan_coonhound': 24,
 '026.Black_russian_terrier': 25,
 '027.Bloodhound': 26,
 '028.Bluetick_coonhound': 27,
 '029.Border_collie': 28,
 '030.Border_terrier': 29,
 '031.Borzoi': 30,
 '032.Boston_terrier': 31,
 '033.Bouvier_des_flandres': 32,
 '034.Boxer': 33,
 '035.Boykin_spaniel': 34,
 '036.Briard': 35,
 '037.Brittany': 36,
 '038.Brussels_griffon': 37,
 '039.Bull_terrier': 38,
 '040.Bulldog': 39,
 '041.Bullmastiff': 40,
 '042.Cairn_terrier': 41,
 '043.Canaan_dog': 42,
 '044.Cane_corso': 43,
 '045.Cardigan_welsh_corgi': 44,
 '046.Cavalier_king_charles_spaniel': 45,
 '047.Chesapeake_bay_retriever': 46,
 '048.Chihuahua': 47,
 '049.Chinese_crested': 48,
 '050.Chinese_shar-pei': 49,
 '051.Chow_chow': 50,
 '052.Clumber_spaniel': 51,
 '053.Cocker_spaniel': 52,
 '054.Collie': 53,
 '055.Curly-coated_retriever': 54,
 '056.Dachshund': 55,
 '057.Dalmatian': 56,
 '058.Dandie_dinmont_terrier': 57,
 '059.Doberman_pinscher': 58,
 '060.Dogue_de_bordeaux': 59,
 '061.English_cocker_spaniel': 60,
 '062.English_setter': 61,
 '063.English_springer_spaniel': 62,
 '064.English_toy_spaniel': 63,
 '065.Entlebucher_mountain_dog': 64,
 '066.Field_spaniel': 65,
 '067.Finnish_spitz': 66,
 '068.Flat-coated_retriever': 67,
 '069.French_bulldog': 68,
 '070.German_pinscher': 69,
 '071.German_shepherd_dog': 70,
 '072.German_shorthaired_pointer': 71,
 '073.German_wirehaired_pointer': 72,
 '074.Giant_schnauzer': 73,
 '075.Glen_of_imaal_terrier': 74,
 '076.Golden_retriever': 75,
 '077.Gordon_setter': 76,
 '078.Great_dane': 77,
 '079.Great_pyrenees': 78,
 '080.Greater_swiss_mountain_dog': 79,
 '081.Greyhound': 80,
 '082.Havanese': 81,
 '083.Ibizan_hound': 82,
 '084.Icelandic_sheepdog': 83,
 '085.Irish_red_and_white_setter': 84,
 '086.Irish_setter': 85,
 '087.Irish_terrier': 86,
 '088.Irish_water_spaniel': 87,
 '089.Irish_wolfhound': 88,
 '090.Italian_greyhound': 89,
 '091.Japanese_chin': 90,
 '092.Keeshond': 91,
 '093.Kerry_blue_terrier': 92,
 '094.Komondor': 93,
 '095.Kuvasz': 94,
 '096.Labrador_retriever': 95,
 '097.Lakeland_terrier': 96,
 '098.Leonberger': 97,
 '099.Lhasa_apso': 98,
 '100.Lowchen': 99,
 '101.Maltese': 100,
 '102.Manchester_terrier': 101,
 '103.Mastiff': 102,
 '104.Miniature_schnauzer': 103,
 '105.Neapolitan_mastiff': 104,
 '106.Newfoundland': 105,
 '107.Norfolk_terrier': 106,
 '108.Norwegian_buhund': 107,
 '109.Norwegian_elkhound': 108,
 '110.Norwegian_lundehund': 109,
 '111.Norwich_terrier': 110,
 '112.Nova_scotia_duck_tolling_retriever': 111,
 '113.Old_english_sheepdog': 112,
 '114.Otterhound': 113,
 '115.Papillon': 114,
 '116.Parson_russell_terrier': 115,
 '117.Pekingese': 116,
 '118.Pembroke_welsh_corgi': 117,
 '119.Petit_basset_griffon_vendeen': 118,
 '120.Pharaoh_hound': 119,
 '121.Plott': 120,
 '122.Pointer': 121,
 '123.Pomeranian': 122,
 '124.Poodle': 123,
 '125.Portuguese_water_dog': 124,
 '126.Saint_bernard': 125,
 '127.Silky_terrier': 126,
 '128.Smooth_fox_terrier': 127,
 '129.Tibetan_mastiff': 128,
 '130.Welsh_springer_spaniel': 129,
 '131.Wirehaired_pointing_griffon': 130,
 '132.Xoloitzcuintli': 131,
 '133.Yorkshire_terrier': 132}


if img is not None:
    if st.button("Predict Breed"):
        with st.spinner('Wait for it...'):
        
            from huggingface_hub import from_pretrained_keras
            from tensorflow.keras.preprocessing import image
            from keras.applications.inception_v3 import preprocess_input, decode_predictions

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
            
            ## Get list of predictions
            pred_dict = dict(zip(index_list, np.round(preds[0]*100,2)))
            Sorted_Prediction_Dictionary = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)

            Count_5Perc = preds[0][preds[0]>0.02]

            if len(Count_5Perc) == 1:
                TopPredictions = Sorted_Prediction_Dictionary[0]
            if len(Count_5Perc) > 1:
                TopPredictions = Sorted_Prediction_Dictionary[0:len(Count_5Perc)]
            
            df = pd.DataFrame.from_dict(TopPredictions, orient='index', columns=['probability'])

            st.table(df)
        
        
        