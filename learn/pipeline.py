import os
import logging
import time

import yaml

from train import Train
from apply_bpe import ApplyBpe
from learn_bpe import LearnBpe
from reader import Reader
from cleanup import CleanUp
from writer import Writer
from binarize import Binarize
from test import Test
from score import Score
from embeds import Embeds

logger = logging.getLogger('pipeline')

class Pipeline:
    def __init__(self, path):
        self.__path = path
        # should be use to create a folder inside jobs/
        self.__timestamp = time.strftime("%Y-%m-%d@%H.%M.%S")
        self.__src_tgt_ext = {}
        self.__group = ''
        
    def __dict_instance(sekf):
        return {
            'apply_bpe': ApplyBpe,
            'learn_bpe': LearnBpe,
            'reader': Reader,
            'cleanup': CleanUp,
            'writer': Writer,
            'binarize': Binarize,
            'train': Train,
            'test': Test,
            'score': Score,
            'embeds': Embeds
        }

    def __parse_yml(self):
        with open(self.__path, 'r') as f:
            parsed = yaml.load(f, Loader=yaml.FullLoader)
        return parsed

    def __parse_description(self):
        parsed = self.__parse_yml()
        return parsed.get('description')

    def __parse_languages(self):
        parsed = self.__parse_yml()
        return parsed.get('languages')
    
    def __parse_group(self):
        parsed = self.__parse_yml()
        return parsed.get('group')

    def __parse_commands(self):
        parsed = self.__parse_yml()
        return parsed.get('pipeline')

    def __mount(self):
        self.__src_tgt_ext = self.__parse_languages()
        # initialize group folder 
        self.__group = self.__parse_group()
        folders = ['bin', 'checkpoints', 'data', 'log', 'tmp']
        path = os.path.abspath(os.path.join('jobs', self.__group, self.__timestamp))
        os.makedirs(path, exist_ok=True)
        logger.info(f"Execution folder `{path}` created.")
        for folder in folders:
            os.mkdir(os.path.join(path, folder))
            logger.info(f'Subdirectory created `{folder}`')
        with open(os.path.join(path, 'description.txt'), 'w') as f:
            f.write(self.__parse_description())
            logger.info("Description file created in `description.txt`")

    def run(self):
        try:
            self.__mount()
            queue = self.__parse_commands()
            instances = self.__dict_instance()
            _input = None
            for command, params in queue.items():
                logger.info(f"Running `{command}` with params {params}")
                if params == 'no_params':
                    step = instances.get(command)(self.__timestamp, self.__group, self.__src_tgt_ext)
                else:
                    step = instances.get(command)(self.__timestamp, self.__group, self.__src_tgt_ext, **params)
                _input = step.routine(_input)
            logger.info("Pipeline ran all steps!")
        except Exception as ex:
            logger.error(ex)
