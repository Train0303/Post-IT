import os
import glob
from PIL import Image
import re

def clean_file_name(x):
    p = re.compile("[0-9]+")
    m = p.findall(x)
    return m[0]

folder_dir = "./gender_classifier/dataset"
categories = os.listdir(folder_dir)

for folder in categories:
    image_dir = folder_dir + '/'+folder
    files = glob.glob(image_dir+"/*.jpg")
    
    for f in files:
        img = Image.open(f)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        #file_name = clean_file_name(f)
        img.save(f+"_trans.jpg")