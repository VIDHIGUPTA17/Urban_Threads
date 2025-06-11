# import streamlit as st
# import os
# st.title("Urban Threads")
# from PIL import Image
#
# def saveuploadfile(uploadfile):
#     try:
#         with open(os.path.join('uploads',uploadfile.name),'wb') as f:
#             f.write(uploadfile.getbuffer())
#         return 1
#     except:
#         return 0
#
# uploadfile=st.file_uploader("Choose and image")
# if uploadfile is not None:
#
#     if saveuploadfile(uploadfile):
#         saved_path = os.path.join("uploads", uploadfile.name)
#         display = Image.open(saved_path)
#         st.image(display)
#
#
#     else:
#         st.header("some error occured in file upload")
#
import cv2
import streamlit as st
import os
from PIL import Image
import pickle
import numpy as np
# from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import tensorflow as tf
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
# import sklearn
import tensorflow.keras
from keras.src.applications.resnet import ResNet50
from tensorflow.keras.preprocessing import image
from cloudservice import upload_image

from tensorflow.keras.layers import GlobalMaxPooling2D
st.title("Urban Threads")
model= ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model=tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D(),
])
featurlist=np.array(pickle.load(open('embeddings.pkl','rb')))
filenames=pickle.load(open('filenames.pkl','rb'))



def extractfeature(path,model):

    img = image.load_img(path, target_size=(224, 224))
    imagearr = image.img_to_array(img)
    expandedimgarray = np.expand_dims(imagearr, axis=0)
    preprocessedimg = preprocess_input(expandedimgarray)
    result = model.predict(preprocessedimg).flatten()
    normresult = result / norm(result)
    return normresult

def recommend(features,featurlist ):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(featurlist)

    dis, indx = neighbors.kneighbors([features])
    return indx


def save_upload(uploaded_file):
    os.makedirs("uploads", exist_ok=True)
    with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    return True

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    if save_upload(uploaded_file):
        saved_path = os.path.join("uploads", uploaded_file.name)
        img = Image.open(saved_path)
        cloud_url = upload_image(saved_path)
        st.image(cloud_url, caption="Uploaded via Cloudinary")
        features=extractfeature(os.path.join("uploads", uploaded_file.name), model)
        indx= recommend(features,featurlist)

        col1,col2,col3,col4,col5=st.columns(5)
        with col1:
            st.image(filenames[indx[0][0]])
        with col2:
            st.image(filenames[indx[0][1]])
        with col3:
            st.image(filenames[indx[0][2]])
        with col4:
            st.image(filenames[indx[0][3]])
        with col5:
            st.image(filenames[indx[0][5]])


    else:
        st.error("File upload failed.")


