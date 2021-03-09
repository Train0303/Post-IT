  
import torch
import argparse

from face_makeup.makeup import hair_dyeing
from face_parser.test import parsing
from SEAN.test import reconstruct

def main(args):
    print(args)
    torch.manual_seed(args.seed)

    if args.mode == 'dyeing':
        if(args.color == None):
            hair_dyeing(respth='./result/dyeing',dspth='./dataset/src')    
        else:
            hair_dyeing(respth='./result/dyeing',dspth='./dataset/src',color=args.color) #test

    elif args.mode == 'refdyeing':
        # Parsing > SEAN
        #face parsing
        parsing(respth='./result/label/refdyeing/src' ,dspth='./dataset/src') # parsing src_image
        parsing(respth='./result/label/refdyeing/ref', dspth='./dataset/ref') # parsing ref_image
        #SEAN
        #reconstruct(args.mode)

    #------------------------------------stargan v2 여기부터--------------------------------------------------
    # elif args.mode == 'styling_ref':
    #     # StarGAN > Parsing > SEAN
    #     args.result_dir = './result/styling_ref
    #     make_img(args) 
    #     parsing(respth='./result/label/styling_ref/src', dspth='./data/src/src') # parsing src_image
    #     parsing(respth='./result/label/styling_ref/ref', dspth='./result/styling_ref') # parsing fake_image
    #     reconstruct(args.mode)

    # elif args.mode == 'styling_rand':
    #     # StarGAN > Parsing > SEAN
    #     args.result_dir = './result/styling_rand
    #     make_img(args)
    #     parsing(respth='./result/label/styling_rand/src' dspth='./data/src/src')
    #     parsing(respth='./result/label/styling_rand/ref', dspth='./result/styling_rand') # parsing fake_image
    #     reconstruct(args.mode)
    #------------------------------------stargan v2 여기까지--------------------------------------------------
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # implement
    parser.add_argument('--mode', type=str, required=True,
                        choices=["dyeing",'refdyeing','styling_ref','styling_rand'], help='mode')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')

    # StarGAN_v2
    parser.add_argument('--img_size', type=int, default=256, help='Image resolution')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers used in DataLoader')

    parser.add_argument('--num_domains', type=int, default=7, help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16, help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64,help='Style code dimension')
    parser.add_argument('--w_hpf', type=float, default=1, help='weight for high-pass filtering')

    parser.add_argument('--resume_iter', type=int, default=100000,help='Iterations to resume training/testing')
    parser.add_argument('--checkpoint_dir', type=str, default='pretrained_network/StarGAN')
    parser.add_argument('--wing_path', type=str, default='pretrained_network/StarGAN/wing.ckpt')

    parser.add_argument('--src_dir', type=str, default='./dataset/gan/src')
    #parser.add_argument('--result_dir', type=str, default='./results/img')
    
    # for styling_ref
    parser.add_argument('--ref_dir', type=str, default='./dataset/gan/ref')

    # for styling_rand
    parser.add_argument('--target_domain', type=int, default=0)
    parser.add_argument('--num_outs_per_domain', type=int, default=1)
    
    # for dyeing
    parser.add_argument('--color', type=int, action='append')
    args = parser.parse_args()
    main(args)