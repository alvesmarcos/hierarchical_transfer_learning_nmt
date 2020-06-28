import subprocess
import os
import re
import unicodedata
import time

import fire

timestamp = time.strftime("%Y-%m-%d.%H.%M.%S")


def create_dirs(corpus, sample=1):
    path = os.path.abspath(os.path.join(
        'dump', corpus, f"data{sample}@{timestamp}"))
    os.makedirs(path, exist_ok=True)
    return path


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    # lowercase, trim and remove non-letter characters [MA]
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def read_corpus(path, source_lang, target_lang, normalize, sample):
    src_path = path + source_lang
    tgt_path = path + target_lang
    src_sentences = list()
    tgt_sentences = list()

    with open(src_path, 'r') as fd:
        for line in fd:
            # line[:-1] in order to remove \n [MA]
            _sentence = normalize_string(line[:-1]) if normalize else line[:-1]
            src_sentences.append(_sentence)
    with open(tgt_path, 'r') as f:
        for line in f:
            # line[:-1] in order to remove \n [MA]
            _sentence = normalize_string(line[:-1]) if normalize else line[:-1]
            tgt_sentences.append(_sentence)
    src_sentences = src_sentences[:sample]
    tgt_sentences = tgt_sentences[:sample]

    # should be same size otherwise you should check the corpus aligned [MA]
    assert len(src_sentences) == len(tgt_sentences)
    return src_sentences, tgt_sentences


def binarize(path, source_lang, target_lang):
    subprocess.call(f"MKL_THREADING_LAYER=GNU fairseq-preprocess \
        --source-lang {source_lang} \
        --target-lang {target_lang} \
        --trainpref \"{path}/train\" --validpref \"{path}/valid\" --testpref \"{path}/test\" \
        --destdir \"{path}\"", shell=True)


def write(path, data):
    with open(path, 'w') as fd:
        for line in data:
            fd.write(f"{line}\n")


def src_tgt_write(data, path, prefix, source_lang, target_lang):
    _tuple = (source_lang, target_lang)
    for lang, ext in zip(data, _tuple):
        write(os.path.join(path, prefix + ext), lang)


def preprocess(corpus, train_prefix, valid_prefix, test_prefix, source_lang, target_lang, sample, normalize=False):
    path = create_dirs(corpus, sample)
    # preprocess data train [MA]
    train = read_corpus(train_prefix, source_lang,
                        target_lang, normalize, sample)
    src_tgt_write(train, path, 'train.', source_lang, target_lang)
    # preprocess data validation [MA]
    valid = read_corpus(valid_prefix, source_lang,
                        target_lang, normalize, sample)
    src_tgt_write(valid, path, 'valid.', source_lang, target_lang)
    # preprocess data test [MA]
    valid = read_corpus(test_prefix, source_lang,
                        target_lang, normalize, sample)
    src_tgt_write(valid, path, 'test.', source_lang, target_lang)
    # this line only runs correctly if all steps above were ok [MA]
    binarize(path, source_lang, target_lang)


if __name__ == "__main__":
    fire.Fire(preprocess)

# to run: [MA]
# python scripts/preprocess.py --corpus=gr-gi --train-prefix=data/libras-train. --valid-prefix=data/libras-valid. --test-prefix=data/libras-test. --source-lang=gr --target-lang=gi --sample=44594 --normalize=False

