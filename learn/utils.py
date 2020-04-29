import os
import unicodedata

import yaml

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def language_ext(lang):
    valid = ['source', 'target']
    if lang not in valid:
        raise ValueError(f"Language must be one of {valid}")
    command = os.path.abspath('command.yml')
    with open(command, 'r') as f:
        parsed = yaml.load(f, Loader=yaml.FullLoader)
    return parsed['languages'][lang]
