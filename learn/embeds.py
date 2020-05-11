import os

import torch
from fairseq.models.lightconv import LightConvModel
import numpy as np

import utils
from step import Step

class Embeds(Step):
    command_name = 'embeds'
    timestamp = None

    def __init__(self, *args, **kwargs):
        self.timestamp = args[0]
        self.group = args[1]
    
    def __load_best_model(self):
        root = os.path.abspath(os.path.join('jobs', self.group, self.timestamp))
        model = LightConvModel.from_pretrained(
            os.path.join(root, 'checkpoints'),
            checkpoint_file='checkpoint_best.pt',
            data_name_or_path=os.path.join(root, 'bin'),
            bpe='subword_nmt',
            bpe_codes=os.path.join(root, 'gen', 'bpe_code.txt')
        )
        return model

    def __write(self, name, embeds, indices, symbols):
        root = os.path.abspath(
            os.path.join('jobs', self.group, self.timestamp, 'checkpoints', name))
        with open(root, 'w') as f:
            for word in symbols:
                #print(f'word: {word}\n')
                lookup_tensor = torch.tensor(indices[word], dtype=torch.long)
                # search embed for word
                word_embed = embeds(lookup_tensor)
            
                word_embed_array = list(word_embed.detach().numpy())
                word_embed_str = str(word_embed_array)[1:-1]
                f.write(word + ': ' + word_embed_str.replace(',', '') + '\n')

    def routine(self, _input):
        model = self.__load_best_model()
        embeds_encoder = model.models[0].encoder.embed_tokens
        embeds_decoder = model.models[0].decoder.embed_tokens

        self.__write('embeds_encoder.txt', embeds_encoder,
            model.src_dict.indices, model.src_dict.symbols)
        self.__write('embeds_decoder.txt', embeds_decoder,
            model.tgt_dict.indices, model.tgt_dict.symbols)
        return None
