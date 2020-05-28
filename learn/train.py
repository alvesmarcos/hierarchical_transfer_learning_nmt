import os
import subprocess
import json
import logging

import utils
from step import Step
from word_embedding import WordEmbedding

logger = logging.getLogger('pipeline')

class Train(Step):
    command_name = 'train'
    timestamp = None    

    def __init__(self, *args, **kwargs):
        logger.info('Train constructed')
        self.__json_params = kwargs['json_params']
        self.__pre_load_embed = kwargs['pre_load_embed']
        self.__strategy = kwargs['strategy']
        self.__embed_folder = kwargs['embed_folder']
        self.timestamp = args[0]
        self.group = args[1]
        self.src_tgt_dic = args[2]
    
    def __adapt_embed(self):
        root = os.path.abspath(os.path.join('jobs', self.group))
        embeds_path = os.path.join(self.__embed_folder, 'checkpoints')
        # process enconder
        embed_encoder_path = os.path.join(root, self.timestamp, 'tmp', 'generated_embeds_encoder.txt')
        word_embedding = WordEmbedding(
            os.path.join(embeds_path, 'embeds_encoder.txt'),
            os.path.join(root, self.timestamp, 'bin', f"dict.{self.src_tgt_dic.get('source')}.txt")
        )
        word_embedding.process_embed(self.__strategy, embed_encoder_path)
        # process decoder
        embed_decoder_path = os.path.join(root, self.timestamp, 'tmp', 'generated_embeds_decoder.txt')
        word_embedding = WordEmbedding(
            os.path.join(embeds_path, 'embeds_decoder.txt'),
            os.path.join(root, self.timestamp, 'bin', f"dict.{self.src_tgt_dic.get('target')}.txt")
        )
        word_embedding.process_embed(self.__strategy, embed_decoder_path)
        return embed_encoder_path, embed_decoder_path


    def routine(self, _input):
        root = os.path.abspath(os.path.join('jobs', self.group, self.timestamp))
        params = utils.extract_params_from_json(self.__json_params)
        output = os.path.join(root, 'checkpoints')
        embed_params = ''

        if self.__pre_load_embed:
            embed_encoder_path, embed_decoder_path = self.__adapt_embed()
            logger.info(f'Load embeds for Encoder: {embed_encoder_path} - Decoder: {embed_encoder_path}')
            embed_params = f"--encoder-embed-path \"{embed_encoder_path}\" --decoder-embed-path \"{embed_decoder_path}\""

        logger.info('Start training...')

        subprocess.call(f"CUDA_VISIBLE_DEVICES=0 fairseq-train \"{_input}\" \
            {params} {embed_params} --save-dir \"{output}\" \
            --tensorboard-logdir \"{os.path.join(root, 'log')}\"", shell=True)
        
        logger.info('Train finished!')
        return output
