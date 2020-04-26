import os

from resource import Resource

BPE_TOKENS = 10000

def learn_bpe(num_operations, corpus, out_file):
    print(
        f"Running learn bpe on {corpus[0]}, {corpus[1]}, {corpus[2]}, {corpus[3]}")
    # concat all train file (source and target)
    os.system(
        f"cat {corpus[0]} {corpus[1]} {corpus[2]} {corpus[3]} > {corpus[4]}")
    # generate table with bpe code
    os.system(
        f"subword-nmt learn-bpe -s {num_operations} < {corpus[4]} > {out_file}")

def apply_bpe(code, corpus, out_file):
    # apply bpe on all files (train and test, source and target)
    for lang, output in zip(corpus, out_file):
        print(f'Running apply bpe on {lang}')
        os.system(f"subword-nmt apply-bpe -c {code} < {lang} > {output}")

if __name__ == "__main__":
    learn_bpe(
        BPE_TOKENS,
        (Resource.DICT_RESOURCE['train+source'],
            Resource.DICT_RESOURCE['valid+source'],
            Resource.DICT_RESOURCE['valid+target'],
            Resource.DICT_RESOURCE['train+target'], 
            Resource.DICT_RESOURCE['train']),
        Resource.DICT_RESOURCE['bpe+code'])

    apply_bpe(
        Resource.DICT_RESOURCE['bpe+code'], 
        (Resource.DICT_RESOURCE['train+source'], Resource.DICT_RESOURCE['train+target'],
            Resource.DICT_RESOURCE['valid+source'], Resource.DICT_RESOURCE['valid+target'],
            Resource.DICT_RESOURCE['test+source'], Resource.DICT_RESOURCE['test+target']),
        (Resource.DICT_RESOURCE['train+source+bpe'], Resource.DICT_RESOURCE['train+target+bpe'],
            Resource.DICT_RESOURCE['valid+source+bpe'], Resource.DICT_RESOURCE['valid+target+bpe'],
            Resource.DICT_RESOURCE['test+source+bpe'], Resource.DICT_RESOURCE['test+target+bpe']))
