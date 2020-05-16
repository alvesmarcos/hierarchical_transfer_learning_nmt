import os
import logging

from sklearn.model_selection import train_test_split

import utils
from step import Step

logger = logging.getLogger('pipeline')

class Writer(Step):
    command_name = 'writer'
    timestamp = None

    def __init__(self, *args, **kwargs):
        logger.info('Writer constructed')
        self.timestamp = args[0]
        self.group = args[1]
        self.src_tgt_dic = args[2]
    
    def __write(self, lang, data):
        path = os.path.abspath(
            os.path.join('jobs', self.group, self.timestamp, 'data', lang + '.txt'))
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
            path = self.__write(ext + self.src_tgt_dic.get('source'), src)
            source_paths.append(path)
            path = self.__write(ext + self.src_tgt_dic.get('target'), tar)
            target_paths.append(path)
        logger.info(f'Output files => Source: {source_paths}, Target: {target_paths}')
        return source_paths, target_paths
