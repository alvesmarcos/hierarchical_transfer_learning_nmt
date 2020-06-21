import fire
from fairseq import options, utils


def run(path):
    embed_dict = utils.parse_embedding(path)
    while True:
        try:
            _input = input('Type a word: ')
            print(embed_dict[_input])
        except:
            print('Sorry this word is not in the embed')


if __name__ == '__main__':
    fire.Fire(run)
