import os
import numpy as np
from keras.preprocessing import image
from keras.applications import densenet
import pickle

model = densenet.DenseNet201(include_top=False, weights='imagenet', input_shape=(256, 256, 3), pooling='avg', classes=1000)


def get_encodings(img):
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    enc_img = densenet.preprocess_input(img)
    enc_img = model.predict(enc_img)
    return enc_img


if __name__ == '__main__':
    names_list = os.listdir("Dataset/images")
    encodings_list = []
    c = 0
    for i in names_list:
        image_path = "./Dataset/images/" + i
        img = image.load_img(image_path, target_size=(256, 256))
        encoding = get_encodings(img)
        encodings_list.append(encoding)
        c += 1
        print(c)
    print(len(names_list), len(encodings_list))
    with open('encodings.txt', 'wb') as file:
        pickle.dump(encodings_list, file)
    with open('enc_names.txt', 'wb') as file:
        pickle.dump(names_list, file)
