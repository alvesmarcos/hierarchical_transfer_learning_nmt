import os
import unicodedata
import json

import yaml

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def offset(data, frac):
    total = len(data)
    return int(total*frac)

def extract_params_from_json(path_json):
    with open(path_json, 'r') as json_file:
        parameters_dict = json.load(json_file)
    str_parameters = ''
    for key, value in parameters_dict[0].items():
        str_parameters += f' {key} {value}'
    return str_parameters