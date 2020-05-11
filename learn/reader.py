import os

import pandas as pd

from step import Step

class Reader(Step):
    command_name = 'reader'
    timestamp = None
    
    def __init__(self, *args, **kwargs):
        self.__path = kwargs['path']
        self.__sample = kwargs['sample']
        self.timestamp = args[0]
        self.group = args[1]
        
    def routine(self, _input=None):
        # read data (corpus) from csv file (header must be ignored)
        df = pd.read_csv(self.__path, sep="\t", header=None)
        # get a sample of corpus
        df = df.sample(frac=self.__sample, random_state=1)
        # save original data
        file = self.__path.split('/')
        df.to_csv(os.path.abspath(os.path.join('jobs', self.group, self.timestamp, 'data', file[-1])))
        return df
