import os
import random

import pandas as pd

from step import Step

class Reader(Step):
    command_name = 'reader'
    timestamp = None
    
    def __init__(self, *args, **kwargs):
        self.__path = kwargs['path']
        self.timestamp = args[0]
        
    def routine(self, _input=None):
        # read data (corpus) from csv file (header must be ignored)
        df = pd.read_csv(self.__path, sep="\t", header=None)
        # reverse phrases from corpus
        reverse = random.choice([True, False])
        df = df.iloc[::-1] if reverse else df
        # save original data
        file = self.__path.split('/')
        df.to_csv(os.path.abspath(os.path.join('jobs', self.timestamp, 'data', file[-1])))
        return df
