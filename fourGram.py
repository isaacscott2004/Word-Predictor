from collections import Counter


import nltk


class FourGram:
    """
    this class is used to predict the most likely next word
    using four-grams .        
    """
    def __init__(self):
        """
        This is the constructor for this class
        it initializes all the values needed to
        calculate the mutual Information of
        each fourgram
        """
        self.brown_words = [word.lower() for word in nltk.corpus.brown.words() if word.isalnum()]
        self.tokens = nltk.word_tokenize(' '.join(self.brown_words))
        self.fourGrams = nltk.ngrams(self.tokens, 4)
        self.token_counts = Counter(self.tokens)
        self.total_tokens = len(self.tokens)
        self.fourGram_counts = Counter(nltk.ngrams(self.tokens, 4))
        self.mutual_I = None

    def _mutual_information(self):
        """
        this method calculates the mutual Information
        for each fourgram and returns a map with the key as
        the fourgram and value as its mutual Information
        :return: a map with the key as a fourgram and
        the value as its mutual information
        """

        if self.mutual_I is None:
            self.mutual_I = {}
            for key in self.fourGrams:
                if len(key) == 4:
                    value = self.fourGram_counts[key]
                    p_abcd = value / self.total_tokens
                    p_a = self.token_counts[key[0]] / self.total_tokens
                    p_b = self.token_counts[key[1]] / self.total_tokens
                    p_c = self.token_counts[key[2]] / self.total_tokens
                    p_d = self.token_counts[key[3]] / self.total_tokens
                    mutual_info = p_abcd / (p_a * p_b * p_c * p_d)
                    self.mutual_I[key] = mutual_info
        return self.mutual_I

    def get_max_p(self, word1, word2, word3):
        """
        this method gets all the fourgrams
        where the first word is equal to word1,
        the second word is equal to word2, and
        the third word is equal to word3
        then it returns the fourth word in the
        fourgram from the fourgram with the largest
        Mutual Information value
        :param word1: first word in a fourgram
        :param word2: second word in a fourgram
        :param word3: the third word in a fourgram
        :return: fourth word in the chosen fourgram
        """
        mutual_information = self._mutual_information()
        max_value = -1
        max_pair = None
        for key, value in mutual_information.items():
            if key[0] == word1 and key[1] == word2 and key[2] == word3:
                if value > max_value:
                    max_value = value
                    max_pair = (key, value)
        if (max_pair is None):
            raise ValueError("There is no Mutual Information for " + word1 + " " + word2 + " " + word3)
        return max_pair
    def getFourGramList(self):
        return self.fourGrams




def main():
    my_fourGram = FourGram()
    print(my_fourGram.get_max_p("get","in", "the"))
    print(type(my_fourGram.getFourGramList()))







if __name__ == "__main__":
    main()
