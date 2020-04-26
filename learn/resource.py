import os

class Resource:
    BASE_FOLDER_GEN = 'data/gen' 
    CORPUS = 'eng-fra.txt'
    TRAIN = 'train.en-fra'
    TEST = 'test.en-fra'
    TRAIN_SOURCE_BPE = 'train.en'
    TRAIN_TARGET_BPE = 'train.fr'
    VALID_SOURCE_BPE = 'valid.en'
    VALID_TARGET_BPE = 'valid.fr'
    TEST_SOURCE_BPE = 'test.en'
    TEST_TARGET_BPE = 'test.fr'
    TRAIN_SOURCE_LANG = 'train+en.txt'
    TRAIN_TARGET_LANG = 'train+fr.txt'
    VALID_SOURCE_LANG = 'valid+en.txt'
    VALID_TARGET_LANG = 'valid+fr.txt'
    TEST_SOURCE_LANG = 'test+en.txt'
    TEST_TARGET_LANG = 'test+fr.txt'
    BPE_CODE = 'bpe+code.txt'
    DICT_RESOURCE = {
        'corpus': os.path.abspath(os.path.join('data', CORPUS)),
        'gen': os.path.abspath(BASE_FOLDER_GEN),
        'train': os.path.abspath(os.path.join(BASE_FOLDER_GEN, TRAIN)),
        'test': os.path.abspath(os.path.join(BASE_FOLDER_GEN, TEST)),
        'train+source': os.path.abspath(os.path.join(BASE_FOLDER_GEN, TRAIN_SOURCE_LANG)),
        'train+target': os.path.abspath(os.path.join(BASE_FOLDER_GEN, TRAIN_TARGET_LANG)),
        'valid+source': os.path.abspath(os.path.join(BASE_FOLDER_GEN, VALID_SOURCE_LANG)),
        'valid+target': os.path.abspath(os.path.join(BASE_FOLDER_GEN, VALID_TARGET_LANG)),
        'test+source': os.path.abspath(os.path.join(BASE_FOLDER_GEN, TEST_SOURCE_LANG)),
        'test+target': os.path.abspath(os.path.join(BASE_FOLDER_GEN, TEST_TARGET_LANG)),
        'bpe+code': os.path.abspath(os.path.join(BASE_FOLDER_GEN, BPE_CODE)),
        'train+source+bpe': os.path.abspath(os.path.join(BASE_FOLDER_GEN, TRAIN_SOURCE_BPE)),
        'train+target+bpe': os.path.abspath(os.path.join(BASE_FOLDER_GEN, TRAIN_TARGET_BPE)),
        'valid+source+bpe': os.path.abspath(os.path.join(BASE_FOLDER_GEN, VALID_SOURCE_BPE)),
        'valid+target+bpe': os.path.abspath(os.path.join(BASE_FOLDER_GEN, VALID_TARGET_BPE)),
        'test+source+bpe': os.path.abspath(os.path.join(BASE_FOLDER_GEN, TEST_SOURCE_BPE)),
        'test+target+bpe': os.path.abspath(os.path.join(BASE_FOLDER_GEN, TEST_TARGET_BPE)),
    }
