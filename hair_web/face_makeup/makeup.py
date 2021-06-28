import cv2
import os
import numpy as np
from skimage.filters import gaussian
from face_makeup.parser import evaluate
import argparse
import glob
import copy

def sharpen(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=5, multichannel=True)

    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)


def hair(image, parsing, part=17, color=[230, 100, 50]):
    b, g, r = color      #[10, 50, 250]       # [10, 250, 10]
    tar_color = np.zeros_like(image)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

    if part == 12 or part == 13:
        image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
    else:
        image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]

    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    if part == 17:
        changed = sharpen(changed)

    changed[parsing != part] = image[parsing != part]
    return changed

def hair_dyeing(respth,dspth,color=[230, 100, 50]):
    cp = './face_makeup/cp/79999_iter.pth'
    files = glob.glob(dspth+'/*.jpg')
    for f in files:
        image = cv2.imread(f)
        image = cv2.resize(image,(512,512))
        parsing = evaluate(f, cp)
        parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)
        image = hair(image, parsing, part=17, color=color)
        print(respth+'/NoReferenceDyeing.jpg')
        cv2.imwrite(respth+'/NoReferenceDyeing.jpg',image)

def hair_change(refpth,srcpth,mode,part=17):
    cp = './face_makeup/cp/79999_iter.pth'
    file1 = glob.glob(srcpth+'/*.jpg')
    file2 = glob.glob(refpth+'/*.jpg')
    for f in file1:
        image1 = cv2.imread(f)
        image1 = cv2.resize(image1,(512,512))
        parsing1 = evaluate(f, cp)
        parsing1 = cv2.resize(parsing1, image1.shape[0:2], interpolation=cv2.INTER_NEAREST)
    for f in file2:
        image2 = cv2.imread(f)
        image2 = cv2.resize(image2,(512,512))
        parsing2 = evaluate(f, cp)
        parsing2 = cv2.resize(parsing2, image2.shape[0:2], interpolation=cv2.INTER_NEAREST)

    for i in range(len(image1)):
        for j in range(len(image1[0])):
            if(parsing1[i][j] == part):
                image1[i][j][0] = 255
                image1[i][j][1] = 255
                image1[i][j][2] = 255
    
    for i in range(len(image2)):
        for j in range(len(image2[0])):
            if(parsing2[i][j] == part):
                image1[i][j][0] = image2[i][j][0]
                image1[i][j][1] = image2[i][j][1]
                image1[i][j][2] = image2[i][j][2]
    cv2.imwrite('./result/'+mode+'/reference_1.jpg',image1)
    

def face_change(refpth,srcpth):
    face_map = [1,2,3,4,5,6,10,11,12,13]
    cp = './face_makeup/cp/79999_iter.pth'
    file1 = glob.glob(refpth+'/*.jpg')
    file2 = glob.glob(srcpth+'/*.jpg')
    
    for f in file1:
        image1 = cv2.imread(f)
        image1 = cv2.resize(image1,(512,512))
        copy_image1 = copy.deepcopy(image1)
        parsing1 = evaluate(f, cp)
        parsing1 = cv2.resize(parsing1, image1.shape[0:2], interpolation=cv2.INTER_NEAREST)
    for f in file2:
        image2 = cv2.imread(f)
        image2 = cv2.resize(image2,(512,512))
        parsing2 = evaluate(f, cp)
        parsing2 = cv2.resize(parsing2, image2.shape[0:2], interpolation=cv2.INTER_NEAREST)

    for i in range(len(image2)):
        for j in range(len(image2[0])):
            if(parsing2[i][j] in face_map):
                image1[i][j][0] = image2[i][j][0]
                image1[i][j][1] = image2[i][j][1]
                image1[i][j][2] = image2[i][j][2]

    for i in range(len(image1)):
        for j in range(len(image1[0])):
            if(parsing1[i][j] == 17):
                image1[i][j][0] = copy_image1[i][j][0]
                image1[i][j][1] = copy_image1[i][j][1]
                image1[i][j][2] = copy_image1[i][j][2]

    cv2.imwrite(refpth+'/reference.jpg',image1)

def remove_background(srcpth,refpth=None):
    cp = './face_makeup/cp/79999_iter.pth'
    file1 = glob.glob(srcpth+'/*')
    for f in file1:
        image1 = cv2.imread(f)
        image1 = cv2.resize(image1,(512,512))
        parsing1 = evaluate(f, cp)
        parsing1 = cv2.resize(parsing1, image1.shape[0:2], interpolation=cv2.INTER_NEAREST)
    if(refpth!=None):
        file2 = glob.glob(refpth+'/*')
        
        for f in file2:
            image2 = cv2.imread(f)
            image2 = cv2.resize(image2,(512,512))
            parsing2 = evaluate(f, cp)
            parsing2 = cv2.resize(parsing2, image2.shape[0:2], interpolation=cv2.INTER_NEAREST)
        for i in range(512):
            for j in range(512):
                if(parsing1[i][j] == 0):
                    image1[i][j][0] = 255
                    image1[i][j][1] = 255
                    image1[i][j][2] = 255
                if(parsing2[i][j] == 0):
                    image2[i][j][0] = 255
                    image2[i][j][1] = 255
                    image2[i][j][2] = 255
        
        cv2.imwrite(file1[0],image1)
        cv2.imwrite(file2[0],image2)
    else:
        for i in range(512):
            for j in range(512):
                if(parsing1[i][j] == 0):
                    image1[i][j][0] = 255
                    image1[i][j][1] = 255
                    image1[i][j][2] = 255
        cv2.imwrite(file1[0],image1)

def latent_processing(refpth,srcpth):
    face_map = [1,2,3,4,5,6,10,11,12,13]
    cp = './face_makeup/cp/79999_iter.pth'
    file1 = glob.glob(refpth+'/*.jpg')
    file2 = glob.glob(srcpth+'/*.jpg')
    
    for f in file1:
        image1 = cv2.imread(f)
        image1 = cv2.resize(image1,(512,512))
        copy_image1 = copy.deepcopy(image1)
        parsing1 = evaluate(f, cp)
        parsing1 = cv2.resize(parsing1, image1.shape[0:2], interpolation=cv2.INTER_NEAREST)

    for f in file2:
        image2 = cv2.imread(f)
        image2 = cv2.resize(image2,(512,512))
        parsing2 = evaluate(f, cp)
        parsing2 = cv2.resize(parsing2, image2.shape[0:2], interpolation=cv2.INTER_NEAREST)

    for i in range(len(image2)):
        for j in range(len(image2[0])):
            if(parsing2[i][j] in face_map):
                image1[i][j][0] = image2[i][j][0]
                image1[i][j][1] = image2[i][j][1]
                image1[i][j][2] = image2[i][j][2]

    for i in range(len(image1)):
        for j in range(len(image1[0])):
            if(parsing1[i][j] == 17):
                image1[i][j][0] = copy_image1[i][j][0]
                image1[i][j][1] = copy_image1[i][j][1]
                image1[i][j][2] = copy_image1[i][j][2]

            if(parsing1[i][j] == 0):
                image1[i][j][0] = 255
                image1[i][j][1] = 255
                image1[i][j][2] = 255

    cv2.imwrite(file1[0],image1)