import json
from PIL import Image
import numpy as np

def save(data, file_path):
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)

def load(file_path):
    with open(file_path) as json_file:
        return json.load(json_file)

def load_json_str(file_path):
    with open(file_path) as json_file:
        return json_file.read()

def load_classes(path):
    with open(path) as file:
        s = file.read()
        return s.split('\n')
        # return [i.split('\t') for i in s]

def load_image(path):
    imag = Image.open(path)
    imag = imag.resize((56, 56))
    a = np.array(imag)
    # consider them as float and normalize
    a = a.astype('float32')
    a /= 255
    return a