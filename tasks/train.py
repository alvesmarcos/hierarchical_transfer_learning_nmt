import os
import subprocess
import json
import time

import fire
import torch
from fairseq.models.lightconv import LightConvModel

from word_embedding import WordEmbedding

timestamp = time.strftime("%Y-%m-%d.%H.%M.%S")


def create_dirs():
    folders = ['checkpoints', 'tmp', 'log', 'embeds']
    path = os.path.abspath(f".train@{timestamp}")
    os.makedirs(path, exist_ok=True)
    for folder in folders:
        os.mkdir(os.path.join(path, folder))
    return path


def adapt_embed(path, embed_path, strategy, source_lang, target_lang):
    # process enconder [MA]
    embed_encoder_path = os.path.join(
        path, 'tmp', 'generated_embeds_encoder.txt')
    word_embedding = WordEmbedding(
        os.path.join(embed_path, 'embeds_encoder.txt'),
        os.path.join(path, 'bin', f"dict.{source_lang}.txt")
    )
    word_embedding.process_embed(strategy, embed_encoder_path)
    # process decoder [MA]
    embed_decoder_path = os.path.join(
        path, 'tmp', 'generated_embeds_decoder.txt')
    word_embedding = WordEmbedding(
        os.path.join(embed_path, 'embeds_decoder.txt'),
        os.path.join(path, 'bin', f"dict.{target_lang}.txt")
    )
    word_embedding.process_embed(strategy, embed_decoder_path)
    return embed_encoder_path, embed_decoder_path


def extract_params_from_json(path_json):
    with open(path_json, 'r') as json_file:
        parameters_dict = json.load(json_file)
    str_parameters = ''
    for key, value in parameters_dict[0].items():
        str_parameters += f' {key} {value}'
    return str_parameters


def load_best_model(path, data_path, checkpoint_name):
    model = LightConvModel.from_pretrained(
        os.path.join(path, 'checkpoints'), checkpoint_file=checkpoint_name, data_name_or_path=os.path.abspath(data_path))
    return model


def write_embeds(path, name, embeds, indices, symbols):
    root = os.path.join(path, 'embeds', name)
    with open(root, 'w') as fd:
        for word in symbols:
            lookup_tensor = torch.tensor(indices[word], dtype=torch.long)
            word_embed = embeds(lookup_tensor)
            word_embed_array = list(word_embed.detach().numpy())
            word_embed_str = str(word_embed_array)[1:-1]
            fd.write(word + ': ' + word_embed_str.replace(',', '') + '\n')


def train(bin_path, fairseq_params, source_lang, target_lang, save_embeds=True, pre_trained=False, embed_path='', strategy=None):
    path = create_dirs()
    params = extract_params_from_json(fairseq_params)
    embed_params = ''

    if pre_trained:
        embed_encoder_path, embed_decoder_path = adapt_embed(
            path, embed_path, strategy, source_lang, target_lang)
        embed_params = f"--encoder-embed-path \"{embed_encoder_path}\" --decoder-embed-path \"{embed_decoder_path}\""

    subprocess.call(f"CUDA_VISIBLE_DEVICES=0 fairseq-train \"{bin_path}\" \
        {params} {embed_params} --save-dir \"{os.path.join(path, 'checkpoints')}\"  \
        --source-lang {source_lang} \
        --target-lang {target_lang} \
        --tensorboard-logdir \"{os.path.join(path, 'log')}\"", shell=True)

    if save_embeds:
        checkpoints_name = ['checkpoint_best.pt', 'checkpoint_last.pt']
        for name in checkpoints_name:
            model = load_best_model(path, bin_path, name)
            embeds_encoder = model.models[0].encoder.embed_tokens
            embeds_decoder = model.models[0].decoder.embed_tokens
            prefix = name.split('_')[1]
            prefix = prefix.split('.')[0]
            write_embeds(path, f'{prefix}_embeds_encoder.txt', embeds_encoder,
                         model.src_dict.indices, model.src_dict.symbols)
            write_embeds(path, f'{prefix}_embeds_decoder.txt', embeds_decoder,
                         model.tgt_dict.indices, model.tgt_dict.symbols)


if __name__ == "__main__":
    fire.Fire(train)

# to run: [MA]
# python tasks/train.py --bin_path=.data100@2020-05-30.06.17.06/ --fairseq_params=params/train.json --source_lang=en --target_lang=fr
