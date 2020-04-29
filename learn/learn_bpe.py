import os
import subprocess

from step import Step

class LearnBpe(Step):
    command_name = 'learn_bpe'
    timestamp = None
    
    def __init__(self, *args, **kwargs):
        self.__bpe_tokens = kwargs['bpe_tokens']
        self.timestamp = args[0]
    
    def routine(self, _input):
        outfile = os.path.abspath(
            os.path.join('jobs', self.timestamp, 'gen', 'bpe_code.txt'))
        tmp_concated = os.path.abspath(os.path.join('jobs', self.timestamp, 'tmp', 'corpus.txt'))
        source_paths, target_paths = _input
        # concat all train and validation (first two paths) files (source and target)
        subprocess.call(
            f"cat {source_paths[0]} {source_paths[1]} {target_paths[0]} {target_paths[1]} > {tmp_concated}", shell=True)
        subprocess.call(
            f"subword-nmt learn-bpe -s {self.__bpe_tokens} < {tmp_concated} > {outfile}", shell=True)
        return _input, outfile
