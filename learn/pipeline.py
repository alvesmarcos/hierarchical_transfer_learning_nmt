import os
from datetime import datetime

import yaml

from apply_bpe import ApplyBpe
from learn_bpe import LearnBpe
from reader import Reader
from selection import Selection
from split import Split
from writer import Writer

class Pipeline:
    def __init__(self, path):
        self.__path = path
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        # should be use to create a folder inside jobs/
        self.__timestamp = str(timestamp)

    def __dict_instance(sekf):
        return {
            'apply_bpe': ApplyBpe,
            'learn_bpe': LearnBpe,
            'reader': Reader,
            'selection': Selection,
            'split': Split,
            'writer': Writer   
        }

    def __parse_commands(self):
        with open(self.__path, 'r') as f:
            parsed = yaml.load(f, Loader=yaml.FullLoader)
        commands = parsed.get('pipeline')
        return commands

    def __mount(self):
        folders = ['bin', 'checkpoints', 'data', 'gen', 'tmp']
        os.mkdir(os.path.abspath(os.path.join('jobs', self.__timestamp)))
        for folder in folders:
            os.mkdir(
                os.path.abspath(os.path.join('jobs', self.__timestamp, folder)))

    def run(self):
        self.__mount()
        queue = self.__parse_commands()
        instances = self.__dict_instance()
        _input = None
        for command, params in queue.items():
            if params == 'no_params':
                step = instances.get(command)(self.__timestamp)
            else:
                step = instances.get(command)(self.__timestamp, **params)
            _input = step.routine(_input)
