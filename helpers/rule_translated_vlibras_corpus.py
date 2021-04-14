import os

import fire
import pandas as pd
from vlibras_translate import translation


def run(path, output):
    path = os.path.abspath(path)
    output = os.path.abspath(output)
    tr = translation.Translation()
    df = pd.read_csv(path)
    gr = list()
    gi = list()
    for _, row in df.iterrows():
        gr.append(tr.rule_translation(row['PT']))
        gi.append(row['GI'])
    assert len(gr) == len(gi)

    with open(output, 'w') as fd:
        for sentence_gr, sentence_gi in zip(gr, gi):
            fd.write(f"{sentence_gr},{sentence_gi}\n")


if __name__ == "__main__":
    fire.Fire(run)

# to run: [MA]
# python helpers/rule_translated_vlibras_corpus.py --path=data/corpus_teste.csv --output=data/name.csv
