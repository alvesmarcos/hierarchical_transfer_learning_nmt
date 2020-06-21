import os

import fire
import pandas as pd


def run(path, output):
    path = os.path.abspath(path)
    output = os.path.abspath(output)
    df = pd.read_csv(path)
    gr = list()
    gi = list()
    for _, row in df.iterrows():
        score = row['Score'].split('%')[0]
        if float(score) > 50:
            gr.append(row['GR'])
            gi.append(row['GI'])
    assert len(gr) == len(gi)
    
    with open(os.path.join(output, 'corpus_vlibras.csv'), 'w') as fd:
        for sentence_gr, sentence_gi in zip(gr, gi):
            fd.write(f'"{sentence_gr}","{sentence_gi}"\n')
    
if __name__ == "__main__":
    fire.Fire(run)

# to run: [MA]
# python helpers/load_fix_vlibras_corpus.py --path=data/full-corpus-analysis.csv --output=data/
