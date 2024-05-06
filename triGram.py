from collections import Counter

import nltk

nltk.download('punkt')
nltk.download('brown')


class TriGram:
    """
    this class is used to predict the most likely next word
    using trigrams.
    """
    def __init__(self):
        """
        This is the constructor for this class
        it initializes all the values needed to
        calculate the mutual Information of
        each trigram
        """
        self.brown_words = [word.lower() for word in nltk.corpus.brown.words() if word.isalnum()]
        self.tokens = nltk.word_tokenize(' '.join(self.brown_words))
        self.trigrams = nltk.trigrams(self.tokens)
        self.token_counts = Counter(self.tokens)
        self.total_tokens = len(self.tokens)
        self.total_trigrams = len(self.trigrams)
        self.trigram_counts = Counter(self.trigrams)
        self.mutual_I = None

    def _mutual_information(self):
        """
        this method calculates the mutual Information
        for each trigram and returns a map with the key as
        the trigram and value as its mutual Information
        :return: a map with the key as a trigram and
        the value as its mutual information
        """
        if self.mutual_I is None:
            self.mutual_I = {}
            for key, value in self.trigram_counts.items():
                if len(key) == 3:
                    p_xyz = value / self.total_trigrams
                    p_x = self.token_counts[key[0]] / self.total_tokens
                    p_y = self.token_counts[key[1]] / self.total_tokens
                    p_z = self.token_counts[key[2]] / self.total_tokens
                    mutual_info = p_xyz / (p_x * p_y * p_z)
                    self.mutual_I[key] = mutual_info
        return self.mutual_I

    def get_max_p(self, word1, word2):
        """
        this method gets all the trigrams
        where the first word is equal to word1
        and the second word is equal to word2
        then it returns the third word in the
        trigram from the trigram with the largest
        Mutual Information value
        :param word1: first word in a trigram
        :param word2: second word in a trigram
        :return: third word in the chosen trigram
        """
        mutual_information = self._mutual_information()
        max_value = -1
        max_pair = None
        for key, value in mutual_information.items():
            if key[0] == word1 and key[1] == word2:
                if value > max_value:
                    max_value = value
                    max_pair = (key, value)
        if max_pair is None:
            raise ValueError("There is no Mutual Information for " + word1 + " " + word2)
        return max_pair


def main():
    my_trigram = TriGram()
    print(my_trigram.get_max_p("a", "banana"))


if __name__ == "__main__":
    main()
