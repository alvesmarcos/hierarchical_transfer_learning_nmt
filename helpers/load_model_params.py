import os

import fire
import torch
from fairseq.models.lightconv import LightConvModel

def load_best_model(path, data_path):
    model = LightConvModel.from_pretrained(
        os.path.join(path, 'checkpoints'), checkpoint_file='checkpoint_best.pt', data_name_or_path=os.path.abspath(data_path))
    print(model[0])

if __name__ == "__main__":
    fire.Fire(load_best_model)