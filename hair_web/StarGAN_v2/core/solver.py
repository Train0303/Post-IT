import os
from os.path import join as ospj
import time
import datetime
from munch import Munch
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from StarGAN_v2.core.model import build_model
from StarGAN_v2.core.checkpoint import CheckpointIO
from StarGAN_v2.core.data_loader import InputFetcher
import StarGAN_v2.core.utils as utils

class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets_ema = build_model(args) 
        checkpoint_dir = args.checkpoint_dir
       
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)
        self.ckptios = [CheckpointIO(checkpoint_dir+'/{:06d}_nets_ema.ckpt', **self.nets_ema)]
        self.to(self.device)

        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name):
                print('Initializing %s...' % name)
                network.apply(utils.he_init)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def gan_start(self,loaders,mode):
        args = self.args
        nets_ema = self.nets_ema
        os.makedirs(args.result_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        if(mode == "ref_styling"):
            src = next(InputFetcher(loaders.src, None, args.latent_dim, 'test'))
            ref = next(InputFetcher(loaders.ref, None, args.latent_dim, 'test'))
            fname = ospj(args.result_dir, 'reference.jpg')
            utils.translate_using_reference(nets_ema,args,src.x,ref.x,ref.y, fname)
        
        elif(mode == "latent_styling"):
            src = next(InputFetcher(loaders.src, None, args.latent_dim, 'test'))
            fname = ospj(args.result_dir, 'latent.jpg')
            device = src.x.device
            trg_domain = args.trg_domain
            N = src.x.size(0)
            y_trg = torch.tensor([trg_domain]*N)
            z_trg_list = torch.randn(1, 1, args.latent_dim).repeat(1, N, 1).to(device)
            utils.translate_using_latent(nets_ema,args,src.x,y_trg,z_trg_list,1,fname)
        print('Working on {}...'.format(fname))