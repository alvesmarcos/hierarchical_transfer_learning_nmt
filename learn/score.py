import os
import subprocess

import utils
from step import Step

class Score(Step):
    command_name = 'score'
    timestamp = None

    def __init__(self, *args, **kwargs):
        self.timestamp = args[0]
        self.group = args[1]
    
    def routine(self, _input):
        root = os.path.abspath(os.path.join('jobs', self.group, self.timestamp))
        target_test = os.path.join(root, 'gen', 'test.fr')
        file_hypo = os.path.join(root, 'tmp', 'hypo.txt')
        score = os.path.join(root, 'tmp', 'score.txt')

        subprocess.call(f"grep ^H \"{_input}\" | cut -f3- > \"{file_hypo}\"", shell=True)
        subprocess.call(f"MKL_THREADING_LAYER=GNU fairseq-score \
                --sys \"{file_hypo}\" --ref \"{target_test}\" | tee \"{score}\"", shell=True)
        return None
