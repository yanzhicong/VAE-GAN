
import os
import sys
import json




def get_config(name):

    with open(os.path.join('cfgs', name + '.json'), 'r') as config_file:
        config_json = json.loads(config_file.read())

    return config_json

    # pass


def print_config(name):
    pass







