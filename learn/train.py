import os
import subprocess
import json

import utils
from step import Step
from word_embedding import WordEmbedding

class Train(Step):
    command_name = 'train'
    timestamp = None    

    def __init__(self, *args, **kwargs):
        self.__json_params = kwargs['json_params']
        self.__pre_load_embed = kwargs['pre_load_embed']
        self.timestamp = args[0]
        self.group = args[1]
    
    def __adapt_embed(self):
        root = os.path.abspath(os.path.join('jobs', self.group))
        dirs = sorted(os.listdir(root))
        embeds_path = os.path.join(root, dirs[-2], 'checkpoints')
        # process enconder
        embed_encoder_path = os.path.join(root, self.timestamp, 'tmp', 'generated_embeds_encoder.txt')
        word_embedding = WordEmbedding(
            os.path.join(embeds_path, 'embeds_encoder.txt'),
            os.path.join(root, self.timestamp, 'bin', f"dict.{utils.language_ext('source')}.txt")
        )
        word_embedding.process_embed('randomly', embed_encoder_path)
        # process decoder
        embed_decoder_path = os.path.join(root, self.timestamp, 'tmp', 'generated_embeds_decoder.txt')
        word_embedding = WordEmbedding(
            os.path.join(embeds_path, 'embeds_decoder.txt'),
            os.path.join(root, self.timestamp, 'bin', f"dict.{utils.language_ext('target')}.txt")
        )
        word_embedding.process_embed('randomly', embed_decoder_path)
        return embed_encoder_path, embed_decoder_path


    def routine(self, _input):
        root = os.path.abspath(os.path.join('jobs', self.group, self.timestamp))
        params = utils.extract_params_from_json(self.__json_params)
        output = os.path.join(root, 'checkpoints')
        embed_params = ''

        if self.__pre_load_embed:
            embed_encoder_path, embed_decoder_path = self.__adapt_embed()
            embed_params = f"--encoder-embed-path \"{embed_encoder_path}\" --decoder-embed-path \"{embed_decoder_path}\""

        subprocess.call(f"CUDA_VISIBLE_DEVICES=0 fairseq-train \"{_input}\" \
            {params} {embed_params} --save-dir \"{output}\" \
            --tensorboard-logdir \"{os.path.join(root, 'log')}\"", shell=True)
        return output
