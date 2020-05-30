import os
import subprocess
import logging

import utils
from step import Step

logger = logging.getLogger('pipeline')

class Binarize(Step):
    command_name = 'binarize'
    timestamp = None    

    def __init__(self, *args, **kwargs):
        logger.info('Binarize constructed')
        self.timestamp = args[0]
        self.group = args[1]
        self.src_tgt_dic = args[2]
    
    def routine(self, _input):
        root = os.path.abspath(os.path.join('jobs', self.group, self.timestamp))
        path = os.path.join(root, 'data')
        output = os.path.join(root, 'bin')
        subprocess.call(f"MKL_THREADING_LAYER=GNU fairseq-preprocess \
            --source-lang {self.src_tgt_dic.get('source')} \
            --target-lang {self.src_tgt_dic.get('target')} \
            --trainpref \"{path}/train\" --validpref \"{path}/valid\" --testpref \"{path}/test\" \
            --destdir \"{output}\"", shell=True)
        logger.info(f'Fairseq binarize generated file in {output} folder')
        return output
