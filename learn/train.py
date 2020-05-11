import os
import subprocess
import json

import utils
from step import Step
from word_embedding import WordEmbedding

class Train(Step):
    command_name = 'train'
    timestamp = None    

    def __init__(self, *args, **kwargs):
        self.__json_params = kwargs['json_params']
        self.timestamp = args[0]
        self.group = args[1]
    
    def __adapt_embed(self):
        raise NotImplementedError

    def routine(self, _input):
        root = os.path.abspath(os.path.join('jobs', self.group, self.timestamp))
        params = utils.extract_params_from_json(self.__json_params)
        output = os.path.join(root, 'checkpoints')
        
        subprocess.call(f"CUDA_VISIBLE_DEVICES=0 fairseq-train {_input} \
            {params} --save-dir {output} \
            --tensorboard-logdir {os.path.join(root, 'log')}", shell=True)
        return output
