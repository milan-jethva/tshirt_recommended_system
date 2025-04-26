import tensorflow
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
import pickle

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False
model = tensorflow.keras.Sequential(
    [
        model,
        GlobalMaxPooling2D()
    ]
)
def extract_feature(img_path,model):
  ima = image.load_img(img_path,target_size=(224,224))
  img_arr = image.img_to_array(ima)
  img_exp = np.expand_dims(img_arr,axis=0)
  preprocessed_img = preprocess_input(img_exp)
  result = model.predict(preprocessed_img).flatten()
  normalized_result = result / norm(result)

  return normalized_result
  
filenames=[]

for file in os.listdir('tshirt'):
  filenames.append(os.path.join('tshirt',file))
  
feature_list = []

for file in filenames:
    feature_list.append(extract_feature(file,model))
  
pickle.dump(feature_list,open('embeddings.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))
