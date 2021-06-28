import torch
import argparse
import shutil
import time
import database
import nvgpu
from face_makeup.makeup import hair_dyeing
from face_makeup.makeup import face_change,remove_background,latent_processing
from face_parser.parser import parsing
from SEAN.sean import SEAN_making
from StarGAN_v2.stargan import img_making
from multiprocessing import Process,Pipe

class maker:
    def __init__(self):
        self.args = InitParser()
        torch.manual_seed(self.args.seed)
    
    def run(self,ID,mode,proc,hair_style,color=[0,0,0]):
        
        if database.gpu_check() == 1:
            while(True):
                time.sleep(1)
                if database.gpu_check() == 0:
                    break

        database.gpu_change(1)
        proc.send(1)
        proc.close()
        result_file = './result/'+ID+'/result/result.png'
        object_file = './static/images/'+ID+'/result.png'

        
        self.args.ID = ID
        self.args.mode = mode
        if self.args.mode == 'RGB_dyeing':
            hair_dyeing(respth='./result/'+self.args.ID+'/RGB_dyeing',dspth='./dataset/'+self.args.ID+'/src/src',color=color)
            remove_background(srcpth='./dataset/'+self.args.ID+'/src/src',refpth='./result/'+self.args.ID+'/RGB_dyeing')
            SEAN_making(self.args.mode,self.args.ID)
                

        elif self.args.mode == 'ref_dyeing':
            remove_background(srcpth='./dataset/'+self.args.ID+'/src/src',refpth='./dataset/'+self.args.ID+'/ref/ref')
            SEAN_making(self.args.mode,self.args.ID)

        
        elif self.args.mode == 'ref_styling':
            self.args.checkpoint_dir = './StarGAN_v2/checkpoints/celeba_hq'
            self.args.wing_path = './StarGAN_v2/checkpoints/wing.ckpt'
            self.args.num_domains = 2
            self.args.src_dir = './dataset/'+self.args.ID+'/src'
            self.args.ref_dir = './dataset/'+self.args.ID+'/ref'
            self.args.result_dir = './result/'+self.args.ID+'/ref_styling'
            
            remove_background(srcpth='./dataset/'+self.args.ID+'/src/src',refpth='./dataset/'+self.args.ID+'/ref/ref')
            img_making(self.args)
            face_change(refpth='./result/'+self.args.ID+'/ref_styling',srcpth='./dataset/'+self.args.ID+'/src/src')
            SEAN_making(self.args.mode,self.args.ID)
            

        elif self.args.mode == 'latent_styling':
            self.args.checkpoint_dir = './StarGAN_v2/checkpoints/custom'
            self.args.wing_path = './StarGAN_v2/checkpoints/wing_custom.ckpt'
            self.args.num_domains = 4
            self.args.trg_domain = hair_style
            self.args.src_dir = './dataset/'+self.args.ID+'/src'
            self.args.ref_dir = './dataset/'+self.args.ID+'/ref'
            self.args.result_dir = './result/'+self.args.ID+'/latent_styling'

            remove_background(srcpth='./dataset/'+self.args.ID+'/src/src')
            img_making(self.args)
            latent_processing(refpth='./result/'+self.args.ID+'/latent_styling',srcpth='./dataset/'+self.args.ID+'/src/src')
            SEAN_making(self.args.mode,self.args.ID)

        else:
            raise NotImplementedError
        shutil.move(result_file,object_file)
        print("RUN Finish")
        
        torch.cuda.empty_cache()
        database.gpu_change(0)

    def runTest(self,ID,mode,color=[0,0,0]):
        
        self.args.ID = ID
        self.args.mode = mode
        if self.args.mode == 'RGB_dyeing':
            hair_dyeing(respth='./result/'+self.args.ID+'/RGB_dyeing',dspth='./dataset/'+self.args.ID+'/src/src',color=color)
            remove_background(srcpth='./dataset/'+self.args.ID+'/src/src',refpth='./result/'+self.args.ID+'/RGB_dyeing')
            SEAN_making(self.args.mode,self.args.ID)
                

        elif self.args.mode == 'ref_dyeing':
            remove_background(srcpth='./dataset/'+self.args.ID+'/src/src',refpth='./dataset/'+self.args.ID+'/ref/ref')
            SEAN_making(self.args.mode,self.args.ID)

        
        elif self.args.mode == 'ref_styling':
            self.args.src_dir = './dataset/'+self.args.ID+'/src'
            self.args.ref_dir = './dataset/'+self.args.ID+'/ref'
            self.args.result_dir = './result/'+self.args.ID+'/ref_styling'
            remove_background(srcpth='./dataset/'+self.args.ID+'/src/src',refpth='./dataset/'+self.args.ID+'/ref/ref')
            img_making(self.args)
            face_change(refpth='./result/'+self.args.ID+'/ref_styling',srcpth='./dataset/'+self.args.ID+'/src/src')
            SEAN_making(self.args.mode,self.args.ID)
            

        elif self.args.mode == 'latent_styling':
            self.args.src_dir = './dataset/'+self.args.ID+'/src'
            self.args.ref_dir = './dataset/'+self.args.ID+'/ref'
            self.args.result_dir = './result/'+self.args.ID+'/latent_styling'
            remove_background(srcpth='./dataset/'+self.args.ID+'/src/src')
            img_making(self.args)
            latent_processing(refpth='./result/'+self.args.ID+'/latent_styling',srcpth='./dataset/'+self.args.ID+'/src/src')
            SEAN_making(self.args.mode,self.args.ID)

        else:
            raise NotImplementedError
        
        print("RUN Finish")
        

        torch.cuda.empty_cache()

    
def InitParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str)
                        
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')
    parser.add_argument('--ID',type=str,default='test')

    # StarGAN_v2
    parser.add_argument('--img_size', type=int, default=256, help='Image resolution')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers used in DataLoader')

    parser.add_argument('--num_domains', type=int, default=2, help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16, help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64,help='Style code dimension')
    parser.add_argument('--w_hpf', type=float, default=1, help='weight for high-pass filtering')

    parser.add_argument('--resume_iter', type=int, default=100000,help='Iterations to resume training/testing')
    parser.add_argument('--checkpoint_dir', type=str, default='./StarGAN_v2/checkpoints/celeba_hq')
    parser.add_argument('--wing_path', type=str, default='./StarGAN_v2/checkpoints/wing.ckpt')

    parser.add_argument('--src_dir', type=str, default='./dataset/src')
    parser.add_argument('--ref_dir', type=str, default='./dataset/ref')


    parser.add_argument('--trg_domain', type=int, default=0)
    parser.add_argument('--num_outs_per_domain', type=int, default=1)

    args = parser.parse_args()
    return args