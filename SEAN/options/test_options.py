"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self, parser,ID):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--result_dir', type=str, default='./result/'+ID+'/result', help='saves results here.')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=float("inf"), help='how many test images to run')
        parser.set_defaults(preprocess_mode='scale_width_and_crop', crop_size=256, load_size=256, display_winsize=256)
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')

        parser.add_argument('--status', type=str, default='test')
        parser.add_argument('--mode', type=str, default='NONE')
        parser.add_argument('--target_domain', type=str, default='NONE')

        self.isTrain = False
        return parser