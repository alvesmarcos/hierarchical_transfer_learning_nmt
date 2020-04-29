from sklearn.model_selection import train_test_split

from step import Step

class Split(Step):
    command_name = 'split'
    timestamp = None

    def __init__(self, *args, **kwargs):
        self.__valid_size = kwargs['valid_size']
        self.__test_size = kwargs['test_size']
    
    def routine(self, _input):
        src_lang, targ_lang = _input
        # split data in train/test 
        X_train, X_test, y_train, y_test = train_test_split(
            src_lang, targ_lang, test_size=self.__test_size)
        # split remaining data in train/valid
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train,test_size=self.__valid_size)
        # tuple with source and target to train/valid/test
        return (X_train, X_valid, X_test), (y_train, y_valid, y_test)
