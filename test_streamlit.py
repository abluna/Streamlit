import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing import image
from keras.applications.resnet_v2 import preprocess_input, decode_predictions

st.write("""
# Dog or Bird Classification Tool
The goal of this tool is to quickly predict a dog's breed or bird species based on a single image. \n
To run a prediction, upload a photo and click 'Predict'
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

    dog_model = keras.saving.load_model('hf://abluna/dog_breed_v2')
    bird_model = keras.saving.load_model('hf://abluna/bird_classification_v4')
    
    return dog_model, bird_model

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

dog_index_list = {'AFFENPINSCHER': 0,
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

bird_index_list = {'Abert_s Towhee Bird': 0,
 'Acorn Woodpecker Bird': 1,
 'Allen_s Hummingbird Bird': 2,
 'American Avocet Bird': 3,
 'American Bittern Bird': 4,
 'American Coot Bird': 5,
 'American Crow Bird': 6,
 'American Dipper Bird': 7,
 'American Golden-Plover Bird': 8,
 'American Goldfinch Bird': 9,
 'American Goshawk Bird': 10,
 'American Kestrel Bird': 11,
 'American Oystercatcher Bird': 12,
 'American Pipit Bird': 13,
 'American Redstart Bird': 14,
 'American Robin Bird': 15,
 'American Tree Sparrow Bird': 16,
 'American White Pelican Bird': 17,
 'American Wigeon Bird': 18,
 'Ancient Murrelet Bird': 19,
 'Anhinga Bird': 20,
 'Anna_s Hummingbird Bird': 21,
 'Arctic Tern Bird': 22,
 'Ash-throated Flycatcher Bird': 23,
 'Ashy Storm-Petrel Bird': 24,
 'Baird_s Sandpiper Bird': 25,
 'Baird_s Sparrow Bird': 26,
 'Bald Eagle Bird': 27,
 'Baltimore Oriole Bird': 28,
 'Band-tailed Pigeon Bird': 29,
 'Bank Swallow Bird': 30,
 'Bar-tailed Godwit Bird': 31,
 'Barn Owl Bird': 32,
 'Barn Swallow Bird': 33,
 'Bay-breasted Warbler Bird': 34,
 'Belcher_s Gull Bird': 35,
 'Bell_s Sparrow Bird': 36,
 'Bell_s Vireo Bird': 37,
 'Belted Kingfisher Bird': 38,
 'Bendire_s Thrasher Bird': 39,
 'Bewick_s Wren Bird': 40,
 'Black Oystercatcher Bird': 41,
 'Black Phoebe Bird': 42,
 'Black Rail Bird': 43,
 'Black Scoter Bird': 44,
 'Black Skimmer Bird': 45,
 'Black Storm-Petrel Bird': 46,
 'Black Swift Bird': 47,
 'Black Tern Bird': 48,
 'Black Turnstone Bird': 49,
 'Black Vulture Bird': 50,
 'Black-and-white Warbler Bird': 51,
 'Black-backed Oriole Bird': 52,
 'Black-bellied Plover Bird': 53,
 'Black-chinned Hummingbird Bird': 54,
 'Black-chinned Sparrow Bird': 55,
 'Black-crowned Night Heron Bird': 56,
 'Black-footed Albatross Bird': 57,
 'Black-headed Grosbeak Bird': 58,
 'Black-headed Gull Bird': 59,
 'Black-legged Kittiwake Bird': 60,
 'Black-necked Stilt Bird': 61,
 'Black-tailed Gnatcatcher Bird': 62,
 'Black-tailed Gull Bird': 63,
 'Black-throated Blue Warbler Bird': 64,
 'Black-throated Gray Warbler Bird': 65,
 'Black-throated Green Warbler Bird': 66,
 'Black-throated Magpie-Jay Bird': 67,
 'Black-throated Sparrow Bird': 68,
 'Black-vented Shearwater Bird': 69,
 'Blackburnian Warbler Bird': 70,
 'Blackpoll Warbler Bird': 71,
 'Blue Grosbeak Bird': 72,
 'Blue-footed Booby Bird': 73,
 'Blue-gray Gnatcatcher Bird': 74,
 'Blue-headed Vireo Bird': 75,
 'Blue-throated Mountain-gem Bird': 76,
 'Blue-winged Teal Bird': 77,
 'Blue-winged Warbler Bird': 78,
 'Bobolink Bird': 79,
 'Bohemian Waxwing Bird': 80,
 'Bonaparte_s Gull Bird': 81,
 'Brandt_s Cormorant Bird': 82,
 'Brant Bird': 83,
 'Brewer_s Blackbird Bird': 84,
 'Brewer_s Sparrow Bird': 85,
 'Bridled Tern Bird': 86,
 'Broad-billed Hummingbird Bird': 87,
 'Broad-tailed Hummingbird Bird': 88,
 'Broad-winged Hawk Bird': 89,
 'Bronzed Cowbird Bird': 90,
 'Brown Booby Bird': 91,
 'Brown Creeper Bird': 92,
 'Brown Pelican Bird': 93,
 'Brown Thrasher Bird': 94,
 'Brown-crested Flycatcher Bird': 95,
 'Brown-headed Cowbird Bird': 96,
 'Buff-breasted Sandpiper Bird': 97,
 'Bufflehead Bird': 98,
 'Buller_s Shearwater Bird': 99,
 'Bullock_s Oriole Bird': 100,
 'Burrowing Owl Bird': 101,
 'Bushtit Bird': 102,
 'Cackling Goose Bird': 103,
 'Cactus Wren Bird': 104,
 'California Condor Bird': 105,
 'California Gnatcatcher Bird': 106,
 'California Gull Bird': 107,
 'California Quail Bird': 108,
 'California Scrub-Jay Bird': 109,
 'California Thrasher Bird': 110,
 'California Towhee Bird': 111,
 'Calliope Hummingbird Bird': 112,
 'Canada Goose Bird': 113,
 'Canada Warbler Bird': 114,
 'Canvasback Bird': 115,
 'Canyon Wren Bird': 116,
 'Cape May Warbler Bird': 117,
 'Caspian Tern Bird': 118,
 'Cassin_s Auklet Bird': 119,
 'Cassin_s Finch Bird': 120,
 'Cassin_s Kingbird Bird': 121,
 'Cassin_s Sparrow Bird': 122,
 'Cassin_s Vireo Bird': 123,
 'Cedar Waxwing Bird': 124,
 'Cerulean Warbler Bird': 125,
 'Chestnut-collared Longspur Bird': 126,
 'Chestnut-sided Warbler Bird': 127,
 'Chimney Swift Bird': 128,
 'Chipping Sparrow Bird': 129,
 'Cinnamon Teal Bird': 130,
 'Cinnamon-rumped Seedeater Bird': 131,
 'Clark_s Grebe Bird': 132,
 'Clark_s Nutcracker Bird': 133,
 'Clay-colored Sparrow Bird': 134,
 'Cliff Swallow Bird': 135,
 'Common Black Hawk Bird': 136,
 'Common Gallinule Bird': 137,
 'Common Goldeneye Bird': 138,
 'Common Grackle Bird': 139,
 'Common Ground Dove Bird': 140,
 'Common Loon Bird': 141,
 'Common Merganser Bird': 142,
 'Common Murre Bird': 143,
 'Common Nighthawk Bird': 144,
 'Common Poorwill Bird': 145,
 'Common Raven Bird': 146,
 'Common Redpoll Bird': 147,
 'Common Tern Bird': 148,
 'Common Yellowthroat Bird': 149,
 'Connecticut Warbler Bird': 150,
 'Cook_s Petrel Bird': 151,
 'Cooper_s Hawk Bird': 152,
 'Cory_s Shearwater Bird': 153,
 'Costa_s Hummingbird Bird': 154,
 'Craveri_s Murrelet Bird': 155,
 'Crested Caracara Bird': 156,
 'Crissal Thrasher Bird': 157,
 'Curlew Sandpiper Bird': 158,
 'Curve-billed Thrasher Bird': 159,
 'Dark-eyed Junco Bird': 160,
 'Dickcissel Bird': 161,
 'Double-crested Cormorant Bird': 162,
 'Downy Woodpecker Bird': 163,
 'Dunlin Bird': 164,
 'Dusky Flycatcher Bird': 165,
 'Dusky Warbler Bird': 166,
 'Dusky-capped Flycatcher Bird': 167,
 'Eared Grebe Bird': 168,
 'Eastern Kingbird Bird': 169,
 'Eastern Phoebe Bird': 170,
 'Eastern Towhee Bird': 171,
 'Eastern Whip-poor-will Bird': 172,
 'Eastern Wood-Pewee Bird': 173,
 'Elegant Tern Bird': 174,
 'Eurasian Collared-Dove Bird': 175,
 'Eurasian Wigeon Bird': 176,
 'European Starling Bird': 177,
 'Evening Grosbeak Bird': 178,
 'Ferruginous Hawk Bird': 179,
 'Flammulated Owl Bird': 180,
 'Flesh-footed Shearwater Bird': 181,
 'Fork-tailed Storm-Petrel Bird': 182,
 'Forster_s Tern Bird': 183,
 'Fox Sparrow Bird': 184,
 'Franklin_s Gull Bird': 185,
 'Fulvous Whistling-Duck Bird': 186,
 'Gadwall Bird': 187,
 'Gambel_s Quail Bird': 188,
 'Gila Woodpecker Bird': 189,
 'Glaucous Gull Bird': 190,
 'Glaucous-winged Gull Bird': 191,
 'Glossy Ibis Bird': 192,
 'Golden Eagle Bird': 193,
 'Golden-crowned Kinglet Bird': 194,
 'Golden-crowned Sparrow Bird': 195,
 'Golden-winged Warbler Bird': 196,
 'Grace_s Warbler Bird': 197,
 'Grasshopper Sparrow Bird': 198,
 'Gray Catbird Bird': 199,
 'Gray Flycatcher Bird': 200,
 'Gray Silky-flycatcher Bird': 201,
 'Gray Thrasher Bird': 202,
 'Gray Vireo Bird': 203,
 'Gray-cheeked Thrush Bird': 204,
 'Gray-faced Petrel Bird': 205,
 'Gray-tailed Tattler Bird': 206,
 'Great Blue Heron Bird': 207,
 'Great Crested Flycatcher Bird': 208,
 'Great Egret Bird': 209,
 'Great Horned Owl Bird': 210,
 'Great Shearwater Bird': 211,
 'Great-tailed Grackle Bird': 212,
 'Greater Pewee Bird': 213,
 'Greater Roadrunner Bird': 214,
 'Greater Scaup Bird': 215,
 'Greater White-fronted Goose Bird': 216,
 'Greater Yellowlegs Bird': 217,
 'Green Heron Bird': 218,
 'Green-tailed Towhee Bird': 219,
 'Green-winged Teal Bird': 220,
 'Groove-billed Ani Bird': 221,
 'Guadalupe Murrelet Bird': 222,
 'Gull-billed Tern Bird': 223,
 'Hairy Woodpecker Bird': 224,
 'Hammond_s Flycatcher Bird': 225,
 'Harlequin Duck Bird': 226,
 'Harris_s Hawk Bird': 227,
 'Harris_s Sparrow Bird': 228,
 'Heermann_s Gull Bird': 229,
 'Hepatic Tanager Bird': 230,
 'Hermit Thrush Bird': 231,
 'Hermit Warbler Bird': 232,
 'Herring Gull Bird': 233,
 'Hooded Merganser Bird': 234,
 'Hooded Oriole Bird': 235,
 'Hooded Warbler Bird': 236,
 'Horned Grebe Bird': 237,
 'Horned Lark Bird': 238,
 'Horned Puffin Bird': 239,
 'House Finch Bird': 240,
 'House Sparrow Bird': 241,
 'House Wren Bird': 242,
 'Hudsonian Godwit Bird': 243,
 'Hutton_s Vireo Bird': 244,
 'Iceland Gull Bird': 245,
 'Inca Dove Bird': 246,
 'Indigo Bunting Bird': 247,
 'Kentucky Warbler Bird': 248,
 'Killdeer Bird': 249,
 'King Eider Bird': 250,
 'Kittlitz_s Murrelet Bird': 251,
 'Ladder-backed Woodpecker Bird': 252,
 'Lapland Longspur Bird': 253,
 'Lark Bunting Bird': 254,
 'Lark Sparrow Bird': 255,
 'Laughing Gull Bird': 256,
 'Lawrence_s Goldfinch Bird': 257,
 'Laysan Albatross Bird': 258,
 'Lazuli Bunting Bird': 259,
 'LeConte_s Sparrow Bird': 260,
 'LeConte_s Thrasher Bird': 261,
 'Leach_s Storm-Petrel Bird': 262,
 'Least Bittern Bird': 263,
 'Least Flycatcher Bird': 264,
 'Least Sandpiper Bird': 265,
 'Least Storm-Petrel Bird': 266,
 'Least Tern Bird': 267,
 'Lesser Black-backed Gull Bird': 268,
 'Lesser Goldfinch Bird': 269,
 'Lesser Nighthawk Bird': 270,
 'Lesser Scaup Bird': 271,
 'Lesser Yellowlegs Bird': 272,
 'Lewis_s Woodpecker Bird': 273,
 'Lilac-crowned Parrot Bird': 274,
 'Lincoln_s Sparrow Bird': 275,
 'Little Blue Heron Bird': 276,
 'Little Bunting Bird': 277,
 'Little Gull Bird': 278,
 'Little Stint Bird': 279,
 'Loggerhead Shrike Bird': 280,
 'Long-billed Curlew Bird': 281,
 'Long-billed Dowitcher Bird': 282,
 'Long-eared Owl Bird': 283,
 'Long-tailed Duck Bird': 284,
 'Long-tailed Jaeger Bird': 285,
 'Louisiana Waterthrush Bird': 286,
 'Lucy_s Warbler Bird': 287,
 'MacGillivray_s Warbler Bird': 288,
 'Magnificent Frigatebird Bird': 289,
 'Magnolia Warbler Bird': 290,
 'Mallard Bird': 291,
 'Mandarin Duck Bird': 292,
 'Manx Shearwater Bird': 293,
 'Marbled Godwit Bird': 294,
 'Marbled Murrelet Bird': 295,
 'Marsh Wren Bird': 296,
 'Masked Booby Bird': 297,
 'Merlin Bird': 298,
 'Mexican Duck Bird': 299,
 'Mexican Whip-poor-will Bird': 300,
 'Mississippi Kite Bird': 301,
 'Mitred Parakeet Bird': 302,
 'Monk Parakeet Bird': 303,
 'Mottled Petrel Bird': 304,
 'Mountain Bluebird Bird': 305,
 'Mountain Chickadee Bird': 306,
 'Mountain Plover Bird': 307,
 'Mountain Quail Bird': 308,
 'Mourning Dove Bird': 309,
 'Mourning Warbler Bird': 310,
 'Nashville Warbler Bird': 311,
 'Nazca Booby Bird': 312,
 'Nelson_s Sparrow Bird': 313,
 'Neotropic Cormorant Bird': 314,
 'Newell_s Shearwater Bird': 315,
 'Northern Flicker Bird': 316,
 'Northern Fulmar Bird': 317,
 'Northern Harrier Bird': 318,
 'Northern Mockingbird Bird': 319,
 'Northern Parula Bird': 320,
 'Northern Pintail Bird': 321,
 'Northern Pygmy-Owl Bird': 322,
 'Northern Red Bishop Bird': 323,
 'Northern Rough-winged Swallow Bird': 324,
 'Northern Saw-whet Owl Bird': 325,
 'Northern Shoveler Bird': 326,
 'Northern Waterthrush Bird': 327,
 'Northern Wheatear Bird': 328,
 'Nuttall_s Woodpecker Bird': 329,
 'Oak Titmouse Bird': 330,
 'Olive-sided Flycatcher Bird': 331,
 'Orange-crowned Warbler Bird': 332,
 'Orchard Oriole Bird': 333,
 'Osprey Bird': 334,
 'Ovenbird Bird': 335,
 'Pacific Golden-Plover Bird': 336,
 'Pacific Loon Bird': 337,
 'Pacific Wren Bird': 338,
 'Painted Bunting Bird': 339,
 'Painted Redstart Bird': 340,
 'Palm Warbler Bird': 341,
 'Parakeet Auklet Bird': 342,
 'Parasitic Jaeger Bird': 343,
 'Pectoral Sandpiper Bird': 344,
 'Pelagic Cormorant Bird': 345,
 'Peregrine Falcon Bird': 346,
 'Phainopepla Bird': 347,
 'Philadelphia Vireo Bird': 348,
 'Pied-billed Grebe Bird': 349,
 'Pigeon Guillemot Bird': 350,
 'Pine Siskin Bird': 351,
 'Pine Warbler Bird': 352,
 'Pink-footed Shearwater Bird': 353,
 'Pinyon Jay Bird': 354,
 'Plumbeous Vireo Bird': 355,
 'Pomarine Jaeger Bird': 356,
 'Prairie Falcon Bird': 357,
 'Prairie Warbler Bird': 358,
 'Prothonotary Warbler Bird': 359,
 'Purple Finch Bird': 360,
 'Purple Gallinule Bird': 361,
 'Purple Martin Bird': 362,
 'Pygmy Nuthatch Bird': 363,
 'Pyrrhuloxia Bird': 364,
 'Red Crossbill Bird': 365,
 'Red Knot Bird': 366,
 'Red Phalarope Bird': 367,
 'Red-billed Tropicbird Bird': 368,
 'Red-breasted Merganser Bird': 369,
 'Red-breasted Nuthatch Bird': 370,
 'Red-breasted Sapsucker Bird': 371,
 'Red-crowned Parrot Bird': 372,
 'Red-eyed Vireo Bird': 373,
 'Red-faced Warbler Bird': 374,
 'Red-footed Booby Bird': 375,
 'Red-lored Parrot Bird': 376,
 'Red-masked Parakeet Bird': 377,
 'Red-naped Sapsucker Bird': 378,
 'Red-necked Grebe Bird': 379,
 'Red-necked Phalarope Bird': 380,
 'Red-necked Stint Bird': 381,
 'Red-shouldered Hawk Bird': 382,
 'Red-tailed Hawk Bird': 383,
 'Red-tailed Tropicbird Bird': 384,
 'Red-throated Loon Bird': 385,
 'Red-throated Pipit Bird': 386,
 'Red-whiskered Bulbul Bird': 387,
 'Red-winged Blackbird Bird': 388,
 'Reddish Egret Bird': 389,
 'Redhead Bird': 390,
 'Rhinoceros Auklet Bird': 391,
 'Ridgway_s Rail Bird': 392,
 'Ring-billed Gull Bird': 393,
 'Ring-necked Duck Bird': 394,
 'Ring-necked Pheasant Bird': 395,
 'Rivoli_s Hummingbird Bird': 396,
 'Rock Pigeon Bird': 397,
 'Rock Wren Bird': 398,
 'Rose-breasted Grosbeak Bird': 399,
 'Rose-ringed Parakeet Bird': 400,
 'Roseate Spoonbill Bird': 401,
 'Ross_s Goose Bird': 402,
 'Rough-legged Hawk Bird': 403,
 'Royal Tern Bird': 404,
 'Ruby-crowned Kinglet Bird': 405,
 'Ruddy Duck Bird': 406,
 'Ruddy Ground Dove Bird': 407,
 'Ruddy Turnstone Bird': 408,
 'Ruff Bird': 409,
 'Rufous Hummingbird Bird': 410,
 'Rufous-backed Robin Bird': 411,
 'Rufous-crowned Sparrow Bird': 412,
 'Rusty Blackbird Bird': 413,
 'Sabine_s Gull Bird': 414,
 'Sage Thrasher Bird': 415,
 'Sagebrush Sparrow Bird': 416,
 'Sanderling Bird': 417,
 'Sandhill Crane Bird': 418,
 'Sandwich Tern Bird': 419,
 'Savannah Sparrow Bird': 420,
 'Say_s Phoebe Bird': 421,
 'Scaly-breasted Munia Bird': 422,
 'Scarlet Tanager Bird': 423,
 'Scissor-tailed Flycatcher Bird': 424,
 'Scott_s Oriole Bird': 425,
 'Scripps_s Murrelet Bird': 426,
 'Semipalmated Plover Bird': 427,
 'Semipalmated Sandpiper Bird': 428,
 'Sharp-shinned Hawk Bird': 429,
 'Sharp-tailed Sandpiper Bird': 430,
 'Short-billed Dowitcher Bird': 431,
 'Short-billed Gull Bird': 432,
 'Short-eared Owl Bird': 433,
 'Short-tailed Albatross Bird': 434,
 'Short-tailed Shearwater Bird': 435,
 'Siberian Sand-Plover Bird': 436,
 'Slaty-backed Gull Bird': 437,
 'Snow Bunting Bird': 438,
 'Snow Goose Bird': 439,
 'Snowy Egret Bird': 440,
 'Snowy Plover Bird': 441,
 'Solitary Sandpiper Bird': 442,
 'Song Sparrow Bird': 443,
 'Sooty Shearwater Bird': 444,
 'Sooty Tern Bird': 445,
 'Sora Bird': 446,
 'South Polar Skua Bird': 447,
 'Spotted Dove Bird': 448,
 'Spotted Owl Bird': 449,
 'Spotted Redshank Bird': 450,
 'Spotted Sandpiper Bird': 451,
 'Spotted Towhee Bird': 452,
 'Sprague_s Pipit Bird': 453,
 'Stejneger_s Petrel Bird': 454,
 'Steller_s Jay Bird': 455,
 'Stilt Sandpiper Bird': 456,
 'Streak-backed Oriole Bird': 457,
 'Streaked Shearwater Bird': 458,
 'Sulphur-bellied Flycatcher Bird': 459,
 'Summer Tanager Bird': 460,
 'Surf Scoter Bird': 461,
 'Surfbird Bird': 462,
 'Swainson_s Hawk Bird': 463,
 'Swainson_s Thrush Bird': 464,
 'Swallow-tailed Gull Bird': 465,
 'Swallow-tailed Kite Bird': 466,
 'Swamp Sparrow Bird': 467,
 'Swinhoe_s White-eye Bird': 468,
 'Tennessee Warbler Bird': 469,
 'Thick-billed Kingbird Bird': 470,
 'Thick-billed Longspur Bird': 471,
 'Townsend_s Solitaire Bird': 472,
 'Townsend_s Storm-Petrel Bird': 473,
 'Townsend_s Warbler Bird': 474,
 'Tree Swallow Bird': 475,
 'Tricolored Blackbird Bird': 476,
 'Tricolored Heron Bird': 477,
 'Tropical Kingbird Bird': 478,
 'Tufted Duck Bird': 479,
 'Tufted Puffin Bird': 480,
 'Tundra Swan Bird': 481,
 'Turkey Vulture Bird': 482,
 'Upland Sandpiper Bird': 483,
 'Varied Bunting Bird': 484,
 'Varied Thrush Bird': 485,
 'Vaux_s Swift Bird': 486,
 'Verdin Bird': 487,
 'Vermilion Flycatcher Bird': 488,
 'Vesper Sparrow Bird': 489,
 'Violet-crowned Hummingbird Bird': 490,
 'Violet-green Swallow Bird': 491,
 'Virginia Rail Bird': 492,
 'Virginia_s Warbler Bird': 493,
 'Wandering Tattler Bird': 494,
 'Warbling Vireo Bird': 495,
 'Warbling White-eye Bird': 496,
 'Wedge-rumped Storm-Petrel Bird': 497,
 'Wedge-tailed Shearwater Bird': 498,
 'Western Bluebird Bird': 499,
 'Western Cattle Egret Bird': 500,
 'Western Flycatcher Bird': 501,
 'Western Grebe Bird': 502,
 'Western Gull Bird': 503,
 'Western Kingbird Bird': 504,
 'Western Meadowlark Bird': 505,
 'Western Sandpiper Bird': 506,
 'Western Screech-Owl Bird': 507,
 'Western Tanager Bird': 508,
 'Western Wood-Pewee Bird': 509,
 'Whimbrel Bird': 510,
 'White Ibis Bird': 511,
 'White Wagtail Bird': 512,
 'White-breasted Nuthatch Bird': 513,
 'White-crowned Sparrow Bird': 514,
 'White-eyed Vireo Bird': 515,
 'White-faced Ibis Bird': 516,
 'White-headed Woodpecker Bird': 517,
 'White-tailed Kite Bird': 518,
 'White-throated Sparrow Bird': 519,
 'White-throated Swift Bird': 520,
 'White-winged Dove Bird': 521,
 'White-winged Scoter Bird': 522,
 'Wild Turkey Bird': 523,
 'Willet Bird': 524,
 'Williamson_s Sapsucker Bird': 525,
 'Willow Flycatcher Bird': 526,
 'Wilson_s Phalarope Bird': 527,
 'Wilson_s Plover Bird': 528,
 'Wilson_s Snipe Bird': 529,
 'Wilson_s Storm-Petrel Bird': 530,
 'Wilson_s Warbler Bird': 531,
 'Winter Wren Bird': 532,
 'Wood Duck Bird': 533,
 'Wood Sandpiper Bird': 534,
 'Wood Stork Bird': 535,
 'Wood Thrush Bird': 536,
 'Worm-eating Warbler Bird': 537,
 'Wrentit Bird': 538,
 'Xantus_s Hummingbird Bird': 539,
 'Yellow Rail Bird': 540,
 'Yellow Warbler Bird': 541,
 'Yellow-bellied Flycatcher Bird': 542,
 'Yellow-bellied Sapsucker Bird': 543,
 'Yellow-billed Cuckoo Bird': 544,
 'Yellow-billed Loon Bird': 545,
 'Yellow-breasted Chat Bird': 546,
 'Yellow-chevroned Parakeet Bird': 547,
 'Yellow-crowned Night Heron Bird': 548,
 'Yellow-footed Gull Bird': 549,
 'Yellow-green Vireo Bird': 550,
 'Yellow-headed Blackbird Bird': 551,
 'Yellow-headed Parrot Bird': 552,
 'Yellow-rumped Warbler Bird': 553,
 'Yellow-throated Vireo Bird': 554,
 'Yellow-throated Warbler Bird': 555,
 'Zone-tailed Hawk Bird': 556}

if option == 'Dog':
    click_predict_message = 'Predict Dog Breed'
else: 
    click_predict_message = 'Predict Bird Species'

if img is not None:
    if st.button(click_predict_message):
        with st.spinner('Wait for it...'):
        
            # Use the function to load your data
            dog_model, bird_model = load_model()

            if option == 'Dog':
                tf_model = dog_model
                index_list = dog_index_list
                targ_size = 350
            else: 
                tf_model = bird_model
                index_list = bird_index_list
                targ_size = 1042
                         
             # `img` is a PIL image of size 224x224
            img_v2 = image.load_img(img, target_size=(targ_size, targ_size))

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



            