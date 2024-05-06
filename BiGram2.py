from collections import Counter

import nltk

nltk.download('punkt')
nltk.download('brown')

class BiGramTwo:
    """
        this class is used to predict the most likely next word
        using bigrams.
        """
    def __init__(self):
        """
                This is the constructor for this class
                It initializes all the values needed to
                calculate the conditional probability of each
                bigram
                """
        self.brown_words = [word.lower() for word in nltk.corpus.brown.words() if word.isalnum()]
        self.tokens = nltk.word_tokenize(' '.join(self.brown_words))
        self.bigrams = nltk.bigrams(self.tokens)
        self.token_counts = Counter(self.tokens)
        self.bi_gram_counts = Counter(self.bigrams)
        self.conditinal_p = None

    def conditional_probability(self):
        """
        this method calculates the conditional probability
        for each bigram and returns a map with the key as
        the bigram and value as its  conditional probability
        :return: a map with the key as bigram and
        the value as its conditional probability
        """
        if self.conditinal_p is None:
            self.conditinal_p = {}
        for key, value in self.bi_gram_counts.items():
            word1 = key[0]
            word1count = self.token_counts[word1]
            self.conditinal_p[key] = value / word1count
        return self.conditinal_p

    def get_max_p(self, word):
        """
        this method gets all the bigrams
        where the first word is equal to word
        then it returns the second word in the
        bigram from bigram with the largest
        conditional probability value
        :param word: first word in a bigram
        :return: second word in the chosen bigram
        """
        conditional_probablity = self.conditional_probability()
        max_value = -1
        max_pair = None
        for key, value in conditional_probablity.items():
            if key[0] == word:
                if value > max_value:
                    max_value = value
                    max_pair = (key, value)
        if max_pair is None:
            raise ValueError("there is no There is no Conditional Probability for " + word)
        return max_pair

def main():
    my_bigram = BiGramTwo()
    print(my_bigram.get_max_p("the"))


if __name__ == "__main__":
    main()