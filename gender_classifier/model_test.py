from keras.models import load_model
import os
import glob
from PIL import Image
import numpy as np
import shutil

model = load_model('gender_classifier.h5',compile=False)

X=list()
data_dir = "./test_data"
result_dir = "./result"
if os.path.exists(result_dir):
    result_categories = os.listdir(result_dir)
else:
    os.makedirs(result_dir)
    os.makedirs(result_dir+"/man")
    os.makedirs(result_dir+"/woman")
    result_categories = os.listdir(result_dir)
    
files = glob.glob(data_dir+"/*.jpg")

for f in files:
    img = Image.open(f)
    img = img.convert("RGB")
    img = img.resize((255,255))
    data = np.asarray(img)
    X.append(data)


input_data = np.asarray(X)
input_data = input_data.astype('float32') / 255
result = model.predict(input_data)

for idx,cat in enumerate(result) :
    if result[idx][0] >= result[idx][1]:
        shutil.move(files[idx],result_dir+"/"+result_categories[0]+'/'+str(idx)+".jpg")
    else:
        shutil.move(files[idx],result_dir+"/"+result_categories[1]+'/'+str(idx)+".jpg")
