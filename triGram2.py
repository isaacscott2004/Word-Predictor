from collections import Counter

import nltk

nltk.download('punkt')
nltk.download('brown')


class TriGramTwo:
    """
    this class is used to predict the most likely next word
    using trigrams.
    """

    def __init__(self):
        """
        This is the constructor for this class
        it initializes all the values needed to
        calculate the conditional probability of
        each trigram
        """
        self.brown_words = [word.lower() for word in nltk.corpus.brown.words() if word.isalnum()]
        self.tokens = nltk.word_tokenize(' '.join(self.brown_words))
        self.trigrams = nltk.trigrams(self.tokens)
        self.bigrams = nltk.bigrams(self.tokens)
        self.trigram_counts = Counter(self.trigrams)
        self.bigram_counts = Counter(self.bigrams)
        self.conditional_p = None

    def _conditional_probability(self):
        """
        this method calculates the conditional probability
        for each trigram and returns a map with the key as
        the trigram and value as its conditional probability
        :return: a map with the key as a trigram and
        the value as its conditional probability
        """
        if self.conditional_p is None:
            self.conditional_p = {}
        for key, value in self.trigram_counts.items():
            bigram = key[:2]
            bigram_count = self.bigram_counts[bigram]
            self.conditional_p[key] = value / bigram_count
        return self.conditional_p

    def get_max_p(self, word1, word2):
        """
        this method gets all the trigrams
        where the first word is equal to word1
        and the second word is equal to word2
        then it returns the third word in the
        trigram from the trigram with the largest
        conditional probability value
        :param word1: first word in a trigram
        :param word2: second word in a trigram
        :return: third word in the chosen trigram
        """
        conditional_probability = self._conditional_probability()
        max_value = -1
        max_pair = None
        for key, value in conditional_probability.items():
            if key[0] == word1 and key[1] == word2:
                if value > max_value:
                    max_value = value
                    max_pair = (key, value)
        if max_pair is None:
            raise ValueError("There is no Conditional Probability for " + word1 + " " + word2)
        return max_pair


def main():
    my_trigram = TriGramTwo()
    print(my_trigram.get_max_p("a", "banana"))


if __name__ == "__main__":
    main()
