import os
import subprocess

import utils
from step import Step

class ApplyBpe(Step):
    command_name = 'apply_bpe'
    timestamp = None

    def __init__(self, *args, **kwargs):
        self.timestamp = args[0]
        self.group = args[1]
    
    def routine(self, _input):
        root = os.path.abspath(os.path.join('jobs', self.group, self.timestamp, 'gen'))
        data = ('train.', 'valid.', 'test.')
        entries, outfile = _input
        source_paths, target_paths = entries
        # apply bpe on all files (train and test, source and target)
        for src_path, tar_path, data_type in zip(source_paths, target_paths, data):
            output_bpe = os.path.join(root, data_type + utils.language_ext('source'))
            subprocess.call(f"subword-nmt apply-bpe -c {outfile} < {src_path} > {output_bpe}", shell=True)
            output_bpe = os.path.join(root, data_type + utils.language_ext('target'))
            subprocess.call(f"subword-nmt apply-bpe -c {outfile} < {tar_path} > {output_bpe}", shell=True)
        return _input
