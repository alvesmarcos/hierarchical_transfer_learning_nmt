import os
import subprocess
import logging

import utils
from step import Step

logger = logging.getLogger('pipeline')

class Test(Step):
    command_name = 'test'
    timestamp = None    

    def __init__(self, *args, **kwargs):
        logger.info('Test constructed')
        self.__json_params = kwargs['json_params']
        self.timestamp = args[0]
        self.group = args[1]
    
    def routine(self, _input):
        root = os.path.abspath(os.path.join('jobs', self.group, self.timestamp))
        _bin = os.path.join(root, 'bin')
        input_test = os.path.join(root, 'data', 'test.'+utils.language_ext('source'))
        ouput = os.path.join(root, 'tmp', 'translated.txt')
        params = utils.extract_params_from_json(self.__json_params)        
        
        logger.info('Start testing...')
        
        subprocess.call(f"MKL_THREADING_LAYER=GNU fairseq-interactive \"{_bin}\" \
            --path \"{_input}/checkpoint_best.pt\" \
            {params} --remove-bpe \
            --tensorboard-logdir \"{os.path.join(root, 'log')}\" < \"{input_test}\" | tee \"{ouput}\"", shell=True)
        
        logger.info('Test finished!')
        return ouput
