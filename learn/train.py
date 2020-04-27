import os

import torch
import torch.nn as nn
from fairseq.models.lightconv import LightConvModel

from resource import Resource

def process(base):
    os.system(f"fairseq-preprocess --source-lang en --target-lang fr \
        --trainpref {base}/train --validpref {base}/valid --testpref {base}/test \
        --destdir bin/tatoeba.tokenized.en-fr")

def train():
    os.system(f"CUDA_VISIBLE_DEVICES=0 fairseq-train bin/tatoeba.tokenized.en-fr \
    --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
    --arch lightconv_iwslt_de_en --save-dir checkpoints/fconv")


def generate():
    os.system(f"fairseq-generate bin/tatoeba.tokenized.en-fr \
    --path checkpoints/fconv/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe")

if __name__ == "__main__":
    #process(Resource.DICT_RESOURCE['gen'])
    #train()
    #generate()
    model = LightConvModel.from_pretrained(
        'checkpoints/fconv',
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path='bin/tatoeba.tokenized.en-fr',
        bpe='subword_nmt',
        bpe_codes='data/gen/bpe+code.txt'
    )
    #print(model.src_dict.indices['hello'])
    lookup_tensor = torch.tensor([model.src_dict.indices['hello']], dtype=torch.long)
    embeds = model.models[0].encoder.embed_tokens
    hello_embed = embeds(lookup_tensor)
    
    print(hello_embed)
    print(embeds)
  
    # print(model.state_dict["embed_tokens"])
    #model.eval()
    # tokens = model.encode(model.src_dict.symbols[10])
    #last_layer_features = model.extract_features(tokens, return_all_hiddens=True)
    #print(last_layer_features)
    # print(tokens)
    # print(model.translate('hello world'))