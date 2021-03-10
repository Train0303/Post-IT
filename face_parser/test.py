#!/usr/bin/python
# -*- encoding: utf-8 -*-
import torch
import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

from face_parser.model import BiSeNet

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
    vis_im = cv2.resize(vis_im,(256,256))
    if save_im:
        print(save_path)
        cv2.imwrite(save_path +'.jpg', vis_im)
        #cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return vis_parsing_anno
    
def parsing(respth, dspth):
    cp = './face_parser/cp/79999_iter.pth'
    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load(cp))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():

        for image_path in os.listdir(dspth):
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0) 
            # rematch label for SEAN implement
            #vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path[:-4]))
            

            """
            0~18  0은 아마 other 논외일 거라고 추측
            """
            parsing[parsing == 0] = 0 
            parsing[parsing == 1] = 1
            parsing[parsing == 2] = 6
            parsing[parsing == 3] = 7
            parsing[parsing == 4] = 4
            parsing[parsing == 5] = 5
            parsing[parsing == 6] = 3
            parsing[parsing == 7] = 8
            parsing[parsing == 8] = 9
            parsing[parsing == 9] = 15
            parsing[parsing == 10] = 2
            parsing[parsing == 11] = 10
            parsing[parsing == 12] = 11
            parsing[parsing == 13] = 12
            parsing[parsing == 14] = 17
            parsing[parsing == 15] = 16
            parsing[parsing == 16] = 18
            parsing[parsing == 17] = 13
            parsing[parsing == 18] = 14
               
            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path[:-4]))
            #vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth,root[-3:],files[0][:-4]))
