from keras.preprocessing import image 
from keras.applications import densenet 
import numpy as np
import pickle
import re
from scipy.spatial.distance import cosine

model = densenet.DenseNet201(include_top=False, weights='imagenet', input_shape=(256, 256, 3), pooling='avg', classes=1000)

with open('C:\\Users\\SAND SPIDER\\Downloads\\project\\encodings.txt', 'rb') as fp:
    enc_list = pickle.load(fp)
with open('C:\\Users\\SAND SPIDER\\Downloads\\project\\enc_names.txt', 'rb') as fp:
    names_list = pickle.load(fp)

def get_encodings(img):
    img = image.img_to_array(img)
    img = image.smart_resize(img, size=(256, 256))
    img = np.expand_dims(img, axis=0)
    enc_img = densenet.preprocess_input(img)
    enc_img = model.predict(enc_img)
    return enc_img.flatten()


def get_recipes(img):
    enc = get_encodings(img)
    similarity_list = []
    recipe_names_list = []
    
    enc_list_flat = [e.flatten() for e in enc_list]
    for i in enc_list_flat:
        similarity = cosine(i, enc)
        similarity_list.append(1 - similarity)
    
    l = sorted(zip(similarity_list, names_list), reverse=True)
    for i in range(len(l)):
        name_in_list = l[i][1]
        s = re.sub(r'[0-9]+.jpg', "", name_in_list)
        if s not in recipe_names_list:
            recipe_names_list.append(s)
        if len(recipe_names_list) >= 10:
            break

    return recipe_names_list
