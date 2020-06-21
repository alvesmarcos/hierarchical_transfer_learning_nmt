import os
import subprocess

import fire

from utils import extract_params_from_json, write_commit_hash


def create_dir(train_path):
    path = os.path.abspath(train_path.replace('train', 'test'))
    os.makedirs(path, exist_ok=True)
    write_commit_hash(os.path.join(path, 'commit.txt'))
    return path


def test(bin_path, train_path, fairseq_params, source_lang, target_lang):
    path = create_dir(train_path)
    src_test_path = os.path.abspath(
        os.path.join(bin_path, 'test.'+source_lang))
    tgt_test_path = os.path.abspath(
        os.path.join(bin_path, 'test.'+target_lang))
    bin_path = os.path.abspath(bin_path)
    train_path = os.path.abspath(train_path)
    params, _ = extract_params_from_json(fairseq_params)

    subprocess.call(f"MKL_THREADING_LAYER=GNU fairseq-interactive \"{bin_path}/\" \
            --path \"{os.path.join(train_path, 'checkpoints')}/checkpoint_best.pt\" \
            {params} \
            --source-lang {source_lang} \
            --target-lang {target_lang} \
            --tensorboard-logdir \"{os.path.join(train_path, 'log')}\" < \"{src_test_path}\" | tee \"{path}/translated.txt\"", shell=True)
    subprocess.call(
        f"grep ^H  \"{path}/translated.txt\" | cut -f3- > \"{path}/hypo.txt\"", shell=True)
    subprocess.call(f"MKL_THREADING_LAYER=GNU fairseq-score \
                --sys \"{path}/hypo.txt\" --ref \"{tgt_test_path}\" | tee \"{path}/score.txt\"", shell=True)


if __name__ == "__main__":
    fire.Fire(test)

# to run:
# python tasks/test.py --bin_path=.data50@2020-06-01.23.40.13/ --train_path=.train__fairseq__50@2020-06-01.23.41.06/ --fairseq_params=params/test.json --source_lang=en --target_lang=fr
