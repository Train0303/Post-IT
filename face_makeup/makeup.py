import cv2
import os
import numpy as np
from skimage.filters import gaussian
from face_makeup.test import evaluate
import argparse
import glob

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
        cv2.imwrite(respth+'/NoReferenceDyeing.jpg',image)