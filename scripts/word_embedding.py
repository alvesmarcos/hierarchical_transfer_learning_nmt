import os

import numpy as np

from text_similarity import TextSimilarity


class WordEmbedding:
    # consts to WordEmbedding
    SHIFT_EMBED = 4
    DIMENSION = 512
    METHODS = ['randomly', 'similarity', 'glove300_pretrained']
    PRE_TRAINED_EMBED_FILE = 'glove.840B.300d.txt'

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
            f.write(f"{len(words)} {self.DIMENSION}\n")
            for word, embed in zip(words, embeds):
                f.write(word + ' ' + embed + '\n')
        return ouput

    def __search_max_min_interval(self, embeds):
        flag = True
        _max = 0
        _min = 0
        for embed in embeds:
            embed_list = [float(x) for x in embed.split()]
            max_embed = max(embed_list)
            min_embed = min(embed_list)
            if flag or max_embed > _max:
                _max = max_embed
            if flag or min_embed < _min:
                _min = min_embed
            flag = False
        return _max, _min

    def __similarity(self, words, embeds, vocabulary, algorithm="levenshtein"):
        words_out = list()
        embeds_out = list()
        for token in vocabulary:
            words_out.append(token)
            if token in words:
                embeds_out.append(embeds[words.index(token)])
            else:
                ts = TextSimilarity(words, embeds, vocabulary)
                # Procura a primeira palavra mais próxima em 0.7 de distancia, max 1.
                embed = ts.search(token)
                embed = embed.split()
                embed_str = str(embed)[1:-1]
                embeds_out.append(embed_str.replace(',', '').replace("'", ''))
        return words_out, embeds_out

    def __randomly(self, words, embeds, vocabulary):
        _max, _ = self.__search_max_min_interval(embeds)
        words_out = list()
        embeds_out = list()
        for token in vocabulary:
            words_out.append(token)
            if token in words:
                embeds_out.append(embeds[words.index(token)])
            else:
                embed = np.random.normal(0, abs(_max), self.DIMENSION).tolist()
                embed_str = str(embed)[1:-1]
                embeds_out.append(embed_str.replace(',', ''))
        return words_out, embeds_out

    def __glove_pre_trained(self, words, embeds, vocabulary):
        words_out = list()
        embeds_out = list()
        words_pre_trained = list()
        embeds_pre_trained = list()
        embed_path = os.path.abspath(os.path.join(
            'data', self.PRE_TRAINED_EMBED_FILE))
        with open(embed_path, 'r') as fd:
            for line in fd:
                content = line.split(' ', 1)
                word, embed = content[0], content[1]
                words_pre_trained.append(word)
                embeds_pre_trained.append(embed[:-1])
        _max, _min = self.__search_max_min_interval(embeds)
        
        for token in vocabulary:
            words_out.append(token)
            if token in words:
                embeds_out.append(embeds[words.index(token)])
            elif token in words_pre_trained:
                index = words_pre_trained.index(token)
                zeros = np.zeros(212).tolist()
                zeros_str = str(zeros)[1:-1]
                embed = embeds_pre_trained[index] + ' ' + zeros_str.replace(',', '')
                embeds_out.append(embed)
            else:
                embed = np.random.uniform(_min, _max, self.DIMENSION).tolist()
                embed_str = str(embed)[1:-1]
                embeds_out.append(embed_str.replace(',', ''))
        return words_out, embeds_out

    def __strategy(self, name, words, embeds, vocabulary):
        if name == 'randomly':
            return self.__randomly(words, embeds, vocabulary)
        elif name == 'similarity':
            return self.__similarity(words, embeds, vocabulary)
        elif name == 'glove300_pretrained':
            return self.__glove_pre_trained(words, embeds, vocabulary)

    def process_embed(self, method, ouput):
        if not method in self.METHODS:
            raise NotImplementedError
        words, embeds, vocabulary = self.__reader()
        header_word, header_embed = words[:self.SHIFT_EMBED], embeds[:self.SHIFT_EMBED]
        words, embeds = self.__strategy(method, words, embeds, vocabulary)
        return self.__write(ouput, header_word + words, header_embed + embeds)
