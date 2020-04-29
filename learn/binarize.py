import os

import utils
from step import Step

class Binarize(Step):
    command_name = 'binarize'
    timestamp = None    

    def __init__(self, *args, **kwargs):
        self.timestamp = args[0]
    
    def routine(self, _input):
        root = os.path.abspath(os.path.join('jobs', self.timestamp))
        path = os.path.join(root, 'gen')
        output = os.path.join(root, 'bin')
        os.system(f"MKL_SERVICE_FORCE_INTEL=1 fairseq-preprocess --source-lang {utils.language_ext('source')} --target-lang {utils.language_ext('target')} \
            --trainpref {path}/train --validpref {path}/valid --testpref {path}/test \
            --destdir {output}")
        return output