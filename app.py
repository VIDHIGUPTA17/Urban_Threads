import image
import tensorflow as tf
import tensorflow.keras
from keras.src.applications.resnet import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
model= ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model=tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D(),
])
print(model.summary())
# print(tf.__version__)
# image.load_img()
def extract_feature(path,model):
    # image= image.load_img
    img = image.load_img(path, target_size=(224, 224))
    imagearr=image.img_to_array(img)
    expandedimgarray=np.expand_dims(imagearr,axis=0)
    preprocessedimg=preprocess_input(expandedimgarray)
    result=model.predict(preprocessedimg).flatten()
    normresult=result / norm(result)
    return normresult



filenames=[]

for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))

featurelist=[]
for file in tqdm(filenames):
    featurelist.append(extract_feature(file,model))


pickle.dump(featurelist,open('embeddings.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))
# print(np.array(featurelist).shape)
