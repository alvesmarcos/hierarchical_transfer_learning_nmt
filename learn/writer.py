import os

from sklearn.model_selection import train_test_split

import utils
from step import Step

class Writer(Step):
    command_name = 'writer'
    timestamp = None

    def __init__(self, *args, **kwargs):
        self.timestamp = args[0]
    
    def __write(self, lang, data):
        path = os.path.abspath(
            os.path.join('jobs', self.timestamp, 'gen', lang + '.txt'))
        with open(path, 'w') as f:
            for line in data:
                f.write(f"{line}\n")
        return path

    def routine(self, _input):
        source_paths = list()
        target_paths = list()
        data = ('train.', 'test.', 'valid.')
        source, target = _input
        for src, tar, ext in zip(source, target, data):
            path = self.__write(ext + utils.language_ext('source'), src)
            source_paths.append(path)
            path = self.__write(ext + utils.language_ext('target'), tar)
            target_paths.append(path)
        return source_paths, target_paths
