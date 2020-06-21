import fire
from sklearn.model_selection import train_test_split


def eng_prefix(line):
    prefix = {
        "i'm": "i am",
        "you're": "you are",
        "it's": "it is",
        "she's": "she is",
        "he's": "he is",
        "we're": "we are",
        "they're": "they are"
    }
    _line = list()
    words = [w.lower() for w in line.split()]
    for w in words:
        _line.append(prefix[w] if w in prefix else w)
    return ' '.join(_line)


def read(path):
    src_sentences = list()
    tgt_sentences = list()
    # reder sentences
    with open(path, 'r') as fd:
        for line in fd:
            source, target = line.split('\t')
            src_sentences.append(eng_prefix(source))
            tgt_sentences.append(target[:-1])  # remove \n
    return src_sentences, tgt_sentences


def split(data_src, data_tgt):
    X_train, X_test, y_train, y_test = train_test_split(
        data_src, data_tgt, test_size=0.3)

    X_test, X_valid, y_test, y_valid = train_test_split(
        X_test, y_test, test_size=0.5
    )
    return (X_train, X_valid, X_test), (y_train, y_valid, y_test)


def write(path, data):
    with open(path, 'w') as fd:
        for d in data:
            fd.write(f'{d}\n')


def run(path):
    source_path = ('data/tatoeba-train.fr-en.en',
                   'data/tatoeba-valid.fr-en.en', 'data/tatoeba-test.fr-en.en')
    target_path = ('data/tatoeba-train.fr-en.fr',
                   'data/tatoeba-valid.fr-en.fr', 'data/tatoeba-test.fr-en.fr')
    source, target = read(path)
    src_data, tgt_data = split(source, target)

    # the length must be the same
    assert len(src_data) == len(tgt_data)

    for src_path, tgt_path, src, tgt in zip(source_path, target_path, src_data, tgt_data):
        write(src_path, src)
        write(tgt_path, tgt)


if __name__ == '__main__':
    fire.Fire(run)
