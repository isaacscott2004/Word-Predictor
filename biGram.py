
from collections import Counter

import nltk

nltk.download('punkt')
nltk.download('brown')


class BiGram:
    """
    this class is used to predict the most likely next word
    using bigrams.
    """


    def __init__(self):
        """
        This is the constructor for this class
        It initializes all the values needed to
        calculate the mutual information of each bigram
        """
        self.brown_words = [word.lower() for word in nltk.corpus.brown.words() if word.isalnum()]
        self.tokens = nltk.word_tokenize(' '.join(self.brown_words))
        self.bigrams = nltk.bigrams(self.tokens)
        self.token_counts = Counter(self.tokens)
        self.total_tokens = len(self.tokens)
        self.total_bigrams = len(self.bigrams)
        self.bi_gram_counts = Counter(self.bigrams)
        self.mutual_I = None

    def _mutual_information(self):
        """
        this method calculates the mutual information
        for each bigram and returns a map with the key as
        the bigram and value as its  mutual information
        :return: a map with the key as bigram and
        the value as its mutual information
        """
        if self.mutual_I is None:
            self.mutual_I = {}
            for key, value in self.bi_gram_counts.items():
                p_xy = value / self.total_bigrams
                p_x = self.token_counts[key[0]] / self.total_tokens
                p_y = self.token_counts[key[1]] / self.total_tokens
                mutual_info = p_xy / (p_x * p_y)
                self.mutual_I[key] = mutual_info
        return self.mutual_I
    def get_max_p(self, word):
        """
        this method gets all the bigrams
        where the first word is equal to word
        then it returns the second word in the
        bigram from this list with the largest
        mutual information value
        :param word: first word in a bigram
        :return: second word in the chosen bigram
        """
        mutual_information = self._mutual_information()
        max_value = -1
        max_pair = None
        for key, value in mutual_information.items():
            if key[0] == word:
                if value > max_value:
                    max_value = value
                    max_pair = (key, value)
        if max_pair is None:
            raise ValueError("there is no Mutual Information for " + word)
        return max_pair





def main():
    my_bigram = BiGram()
    print(my_bigram.get_max_p("the"))


if __name__ == "__main__":
    main()

#