import os
import subprocess
from glob import glob

import utils
from step import Step

class Train(Step):
    command_name = 'train'
    timestamp = None    

    def __init__(self, *args, **kwargs):
        self.__arch = kwargs['arch']
        self.__lr = kwargs['lr']
        self.__dropout = kwargs['dropout']
        self.__max_token = kwargs['max_token']
        self.__patience = kwargs['patience']
        self.__clip_norm = kwargs['clip_norm']
        self.__max_epoch = kwargs['max_epoch']
        self.timestamp = args[0]
    
    def __adapt_embed(self):
        pass
        
    def routine(self, _input):
        root = os.path.abspath(os.path.join('jobs', self.timestamp))
        output = os.path.join(root, 'checkpoints')
        subprocess.call(f"CUDA_VISIBLE_DEVICES=0 fairseq-train {_input} \
            --lr {self.__lr} --clip-norm {self.__clip_norm} --max-epoch {self.__max_epoch} \
            --dropout {self.__dropout} --max-tokens {self.__max_token} \
            --arch {self.__arch}  --save-dir {output} \
            --tensorboard-logdir {os.path.join(root, 'log')}", shell=True)
        return output
