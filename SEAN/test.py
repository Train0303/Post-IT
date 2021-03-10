import os
from collections import OrderedDict


from SEAN import data
from SEAN.options.test_options import TestOptions
from SEAN.models.pix2pix_model import Pix2PixModel
from SEAN.util.visualizer import Visualizer

def reconstruct(mode):
    opt = TestOptions().parse()
    opt.status = 'test'
    opt.contain_dontcare_label = True
    opt.no_instance = True

    if mode == 'dyeing':
        opt.image_dir = './dataset/src'
        opt.label_dir = './result/label/dyeing/src'

    elif mode == 'refdyeing':
        opt.styling_mode = 'dyeing' 

        opt.image_dir = './dataset/src'  #이미지 있는곳
        opt.label_dir = './result/label/refdyeing/src' #라벨 있는곳?

    elif mode == 'styling_ref': # styling_ref
        opt.styling_mode = 'styling'

        opt.image_dir = './dataset/src'
        opt.label_dir = './result/label/styling_ref/src'
    
    else:                       #styling_rand
        opt.styling_mode = 'styling'

        opt.image_dir = './dataset/src'
        opt.label_dir = './result/label/styling_rand/src'
    
    model = Pix2PixModel(opt)
    model.eval()

    visualizer = Visualizer(opt)

    # make dataloader for source image
    src_dataloader = data.create_dataloader(opt)

    if mode == 'dyeing':
        opt.styling_mode = mode

        opt.image_dir = './dataset/dyeing'
        opt.label_dir = './result/label/dyeing/ref'

    elif mode == 'refdyeing':
        opt.styling_mode = 'dyeing' 

        opt.image_dir = './dataset/dyeing'  #이미지 있는곳
        opt.label_dir = './result/label/refdyeing' #라벨 있는곳?

    elif mode == 'styling_ref': # styling_ref
        opt.styling_mode = 'styling'

        opt.image_dir = './result/styling_ref'
        opt.label_dir = './result/label/styling_ref'
    
    else:                       #styling_rand
        opt.styling_mode = 'styling'

        opt.image_dir = './result/styling_rand'
        opt.label_dir = './result/label/styling_rand'


    oth_dataloader = data.create_dataloader(opt)

    for i, data_i in enumerate(zip(src_dataloader,oth_dataloader)):
        src_data = data_i[0]
        oth_data = data_i[1]
        generated = model(src_data,oth_data, mode=opt.styling_mode)
        img_path = src_data['path']

        for b in range(generated.shape[0]):
            print('process image... %s' % img_path[b])
            visuals = OrderedDict([('input_label', data_i['label'][b]),
                                   ('synthesized_image', generated[b])])
            visualizer.save_images(visuals, img_path[b:b+1],opt.results_dir,f'results_{i}')
"""
    for i, data_i in enumerate(zip(cycle(src_dataloader),oth_dataloader)):
        src_data = data_i[0]
        oth_data = data_i[1]
        generated = model(src_data,oth_data, mode=opt.styling_mode)

        img_path = src_data['path']

        for b in range(generated.shape[0]):
            print('process image... %s' % img_path[b])
            visuals = OrderedDict([('input_label', src_data['label'][b]),
                               ('synthesized_image', generated[b])])

            visualizer.save_images(visuals, img_path[b:b + 1],opt.results_dir,f'results_{i}')
    """








    