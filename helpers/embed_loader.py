import argparse

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
    parser = argparse.ArgumentParser(
        description='Test if the embed was load correctly')
    # add arguments
    parser.add_argument('path', type=str, help='The path of the Embed file')
    # process args
    args = parser.parse_args()
    # run script
    run(args.path)
