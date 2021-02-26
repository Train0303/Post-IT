# from keras.preprocessing import image
import os
import glob
from PIL import Image
import numpy as np

folder_dir = "./dataset"
X=list()
Y=list()
categories = os.listdir(folder_dir)
num_class = len(categories)
image_w = 255
image_h = 255
pixels = 255*255*3

for idx,cat in enumerate(categories):
    label = [0 for i in range(num_class)]
    label[idx] = 1
    
    image_dir = folder_dir+'/'+categories[idx]
    files = glob.glob(image_dir+"/*.jpg") 
    
    for i,f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w,image_h))
        data = np.asarray(img)
        X.append(data)
        Y.append(label)

np.save('data.npy',X)
np.save('label.npy',Y)