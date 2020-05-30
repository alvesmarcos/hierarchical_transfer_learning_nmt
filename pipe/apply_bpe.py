import os
import subprocess
import logging

import utils
from step import Step

logger = logging.getLogger('pipeline')

class ApplyBpe(Step):
    command_name = 'apply_bpe'
    timestamp = None

    def __init__(self, *args, **kwargs):
        logger.info('ApplyBpe constructed')
        self.timestamp = args[0]
        self.group = args[1]
        self.src_tgt_dic = args[2]
    
    def routine(self, _input):
        root = os.path.abspath(os.path.join('jobs', self.group, self.timestamp, 'data'))
        data = ('train.', 'valid.', 'test.')
        entries, outfile = _input
        source_paths, target_paths = entries
        # apply bpe on all files (train and test, source and target)
        for src_path, tar_path, data_type in zip(source_paths, target_paths, data):
            output_bpe = os.path.join(root, data_type + self.src_tgt_dic.get('source'))
            subprocess.call(f"subword-nmt apply-bpe -c \"{outfile}\" < \"{src_path}\" > \"{output_bpe}\"", shell=True)
            logger.info(f'BPE applied successfully for source, output: {output_bpe}')
            output_bpe = os.path.join(root, data_type + self.src_tgt_dic.get('target'))
            subprocess.call(f"subword-nmt apply-bpe -c \"{outfile}\" < \"{tar_path}\" > \"{output_bpe}\"", shell=True)
            logger.info(f'BPE applied successfully for target, output: {output_bpe}')
        return _input
