import os

import numpy as np

class WordEmbedding:
    # consts to WordEmbedding
    SHIFT_EMBED = 4
    DIMENSION = 512
    METHODS = ['randomly']

    def __init__(self, embed_file, dict_file):
        self.embed_file = embed_file
        self.dict_file = dict_file

    def __reader(self):
        vocabulary = list()
        words = list()
        embeds = list()
        with open(self.embed_file, 'r') as f:
            for line in f:
                # expected content => je: -0.022472382 0.018156923 -0.02239776
                content = line.split(': ')
                word, embed = content[0], content[1]
                words.append(word)
                embeds.append(embed[:-1])
        with open(self.dict_file, 'r') as f:
            for line in f:
                content = line.split()
                vocabulary.append(content[0])
        return words, embeds, vocabulary

    def __write(self, ouput, words, embeds):
        with open(ouput, 'w') as f:
            for word, embed in zip(words, embeds):
                f.write(word + ': ' + embed + '\n')
        return ouput

    def __randomly(self, words, embeds, vocabulary):
        words_out = list()
        embeds_out = list()
        for token in vocabulary:
            words_out.append(token)
            if token in words:
                embeds_out.append(embeds[words.index(token)])
            else:
                embed = np.random.normal(-0.5, 0.5, self.DIMENSION).tolist()
                embed_str = str(embed)[1:-1]
                embeds_out.append(embed_str.replace(',', ''))
        return words_out, embeds_out                

    def __strategy(self, name, words, embeds, vocabulary):
        if name == 'randomly':
            return self.__randomly(words, embeds, vocabulary)

    def process_embed(self, method, ouput):
        if not method in self.METHODS:
            raise NotImplementedError
        words, embeds, vocabulary = self.__reader()
        header_word, header_embed = words[:self.SHIFT_EMBED], embeds[:self.SHIFT_EMBED]
        words, embeds = self.__strategy(method, words, embeds, vocabulary)
        return self.__write(ouput, header_word + words, header_embed + embeds)

# root = os.path.abspath('jobs')
# dirs = sorted(os.listdir(root))
# embeds_path = os.path.join(root, '1588911091.97093', 'checkpoints')
# # print(embeds_path)

# word_embedding = WordEmbedding(
#     os.path.join(embeds_path, 'embeds_encoder.txt'),
#     os.path.join(root, '1588911678.820784', 'bin', 'dict.en.txt')
# )
# word_embedding.process_embed('randomly', 'generated_embed.txt')
