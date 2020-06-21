import subprocess
import json


def write_commit_hash(path):
    with open(path, 'w') as fd:
        fd.write(subprocess.check_output(
            ["git", "describe", "--always"]).strip().decode())


def extract_params_from_json(path_json):
    with open(path_json, 'r') as json_file:
        parameters_dict = json.load(json_file)
    str_parameters = ''
    for key, value in parameters_dict[0].items():
        str_parameters += f' {key} {value}'
    return str_parameters, parameters_dict[0]
