import os
import subprocess

import utils
from step import Step

class Test(Step):
    command_name = 'test'
    timestamp = None    

    def __init__(self, *args, **kwargs):
        self.__beam = kwargs['beam']
        self.timestamp = args[0]
    
    def routine(self, _input):
        root = os.path.abspath(os.path.join('jobs', self.timestamp))
        _bin = os.path.join(root, 'bin')
        input_test = os.path.join(root, 'gen', 'test.'+utils.language_ext('source'))
        ouput = os.path.join(root, 'tmp', 'translated.txt')
        subprocess.call(f"MKL_SERVICE_FORCE_INTEL=1 fairseq-interactive {_bin} \
            --path {_input}/checkpoint_best.pt \
            --beam {self.__beam} --remove-bpe \
            --tensorboard-logdir {os.path.join(root, 'log')} < {input_test} | tee {ouput}", shell=True)
        return ouput
