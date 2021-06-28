import os
from collections import OrderedDict


from SEAN import data
from SEAN.options.test_options import TestOptions
from SEAN.models.pix2pix_model import Pix2PixModel
from SEAN.util.visualizer import Visualizer
from face_parser.parser import parsing


def SEAN_making(mode,ID):
    opt = TestOptions().parse(ID)
    opt.status = 'test'
    opt.contain_dontcare_label = True
    opt.no_instance = True
    opt.mode = mode
    model = Pix2PixModel(opt)
    model.eval()
    visualizer = Visualizer(opt)

    opt.image_dir, opt.label_dir = src_mode_path(opt,mode,ID)
    src_dpath= opt.image_dir
    src_label = parsing(opt.label_dir,src_dpath)
    src_dataloader = data.create_dataloader(opt)
    
    
    opt.image_dir, opt.label_dir = ref_mode_path(opt,mode,ID)
    ref_dpath = opt.image_dir
    ref_label = parsing(opt.label_dir,ref_dpath)
    ref_dataloader = data.create_dataloader(opt)


    for i, data_i in enumerate(zip(src_dataloader,ref_dataloader)):
        data_i[0]['label']= src_label
        data_i[1]['label']= ref_label
        generated = model(data_i[0],data_i[1], mode=opt.mode)
        img_path = data_i[0]['path']
    
        for b in range(generated.shape[0]):
            print('process image... %s' % img_path[b])
            visuals = OrderedDict({'result': generated[b]})
            visualizer.save_images(visuals, img_path[b:b+1],opt.result_dir)
    

def src_mode_path(opt,mode,ID):
    if mode == 'RGB_dyeing':
        opt.image_dir = './dataset/'+ID+'/src/src'
        opt.label_dir = './result/'+ID+'/label/RGB_dyeing/src'

    elif mode == 'ref_dyeing':
        opt.image_dir = './dataset/'+ID+'/src/src'
        opt.label_dir = './result/'+ID+'/label/ref_dyeing/src'

    elif mode == 'ref_styling':
        opt.image_dir = './result/'+ID+'/ref_styling'
        opt.label_dir = './result/'+ID+'/label/ref_styling/ref'

    else:
        opt.image_dir = './result/'+ID+'/latent_styling'
        opt.label_dir = './result/'+ID+'/label/latent_sytling/ref'

    return opt.image_dir, opt.label_dir

def ref_mode_path(opt,mode,ID):
    if mode == 'RGB_dyeing':
        opt.image_dir = './result/'+ID+'/RGB_dyeing'
        opt.label_dir = './result/'+ID+'/label/dyeing/ref'

    elif mode == 'ref_dyeing':
        opt.image_dir = './dataset/'+ID+'/ref/ref'
        opt.label_dir = './result/'+ID+'/label/refdyeing/ref'

    elif mode == 'ref_styling':
        opt.image_dir = './dataset/'+ID+'/src/src'
        opt.label_dir = './result/'+ID+'/label/ref_styling/src'
    
    else:
        opt.image_dir = './dataset/'+ID+'/src/src'
        opt.label_dir = './result/'+ID+'/label/latent_styling/src'
        
    
    return opt.image_dir, opt.label_dir
