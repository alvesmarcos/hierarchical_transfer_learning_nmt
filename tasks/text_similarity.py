# code created by @withoutCoffee
import random

import textdistance


class TextSimilarity:
    def __init__(self, words, embeds, vocabulary, algorithm="levenshtein"):
        self.words = words
        self.embeds = embeds
        self.vocabulary = vocabulary

        """
        algorithms:
            levenshtein
            jaccard
            ratcliff_obershelp
        """
        self.algorithm = algorithm.lower()

    def search(self, word, limit=0.7):
        """[Pesquisa palavra próxima no vocabulário utilizando um valor de aceitação de distancia.]

        Arguments:
            word {[string]} -- [Palavra de entrada]

        Keyword Arguments:
            limit {float} -- [Valor de distancia para aceitação de palavra (0,1)] (default: {0.7})

        Returns:
            [new_embed] -- []
        """
        dist = list()
        for index, token in enumerate(self.words):
            if self.algorithm == "levenshtein":
                distance = textdistance.levenshtein.normalized_similarity(
                    word, token)
                if distance > limit:
                    return self.embeds[index]
                dist.append(distance)
            elif self.algorithm == "jaccard":
                distance = textdistance.jaccard(word, token)
                if distance > limit:
                    return self.embeds[index]
                dist.append(distance)
            elif self.algorithm == "ratcliff_obershelp":
                distance = textdistance.ratcliff_obershelp(word, token)
                if distance > limit:
                    return self.embeds[index]
                dist.append(distance)
        index = dist.index(max(dist))
        return self.embeds[index]

    def knn_search(self, word):
        """[Pesquisa uma palavra próxima no vocabulário utilizado a lógica do KNN]

        Arguments:
            word {[string]} -- [Palavra de entrada]

        Returns:
            [int] -- [Índice da palava mais próxima no vocabulário]
        """
        dist = list()
        for token in self.vocabulary:
            if self.algorithm == "levenshtein":
                dist.append(
                    textdistance.levenshtein.normalized_similarity(word, token))

            elif self.algorithm == "jaccard":
                dist.append(textdistance.jaccard(word, token))

            elif self.algorithm == "ratcliff_obershelp":
                dist.append(textdistance.ratcliff_obershelp(word, token))

        index = dist.index(max(dist))

        return self.embeds[index]
