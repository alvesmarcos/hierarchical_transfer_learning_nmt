import fire
import pandas as pd
from sklearn.model_selection import train_test_split


def split(data_src, data_tgt):
    X_train, X_test, y_train, y_test = train_test_split(
        data_src, data_tgt, test_size=0.085)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.097)
    return [X_train, X_valid, X_test], [y_train, y_valid, y_test]


def read(path):
    src_sentences = list()
    tgt_sentences = list()
    # reder sentences
    df = pd.read_csv(path, header=None)
    for _, row in df.iterrows():
        src_sentences.append(row[0])
        tgt_sentences.append(row[1])
    return src_sentences, tgt_sentences


def write(path, data):
    with open(path, 'w') as fd:
        for d in data:
            fd.write(f'{d}\n')


def run(path_train_valid, path_test, prefix_src, prefix_tgt):
    base_path = ('data/librasV11-2-train.',
                 'data/librasV11-2-valid.', 'data/librasV11-2-test.')
    source_path = [name + prefix_src for name in base_path]
    target_path = [name + prefix_tgt for name in base_path]
    source, target = read(path_train_valid)
    src_data, tgt_data = split(source, target)
    # source, target = read(path_test)
    src_data.append(source)
    tgt_data.append(target)

    # the length must be the same
    assert len(src_data) == len(tgt_data)

    for src_path, tgt_path, src, tgt in zip(source_path, target_path, src_data, tgt_data):
        write(src_path, src)
        write(tgt_path, tgt)

if __name__ == '__main__':
    fire.Fire(run)
