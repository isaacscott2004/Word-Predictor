import string

import nltk

from WordPredictor.BiGram2 import BiGramTwo
from WordPredictor.biGram import BiGram
from WordPredictor.triGram import TriGram
from WordPredictor.fourGram import FourGram
from WordPredictor.triGram2 import TriGramTwo

nltk.download('punkt')
nltk.download('brown')


def main():
    """
    This is the main of my program. It runs everything
    """
    global final, bi_gram_last_element, tri_gram_last_element, four_gram_last_element, list_of_words
    # my_bigram = BiGram() # mutual information
    my_bigram = BiGramTwo()  # conditional probability
    # my_trigram = TriGram() # mutual information
    my_trigram = TriGramTwo()  # conditional probability
    my_fourgram = FourGram()  # mutual information
    print("Welcome to my word predictor!")
    count = 0
    while True:
        list_of_words = []
        if count > 0:
            otherDecision = input("press 1 to continue your sentence and 2 to start a new sentence or quit the "
                                  "program.\n")
            while otherDecision not in ["1", "2"]:
                otherDecision = input("press 1 to continue your sentence and 2 to start a new sentence or quit the"
                                      " program.\n")
            if otherDecision == "1":
                list_of_words = final.split()
            elif otherDecision == "2":
                list_of_words = []

        sentence = input("Enter some words or type 'quit' to end the program.\n")
        if sentence == "quit":
            break
        bigramError = False
        trigramError = False
        fourgramError = False

        for word in sentence.split():
            word.strip(string.punctuation)
            list_of_words.append(word.lower())
        num_for_indexes = len(list_of_words) - 1
        try:
            bi_gram_prediction = my_bigram.get_max_p(list_of_words[num_for_indexes])
            bi_gram_last_element = bi_gram_prediction[0][-1]
        except ValueError:
            bigramError = True
        except IndexError:
            bigramError = True
        try:
            tri_gram_prediction = my_trigram.get_max_p(list_of_words[num_for_indexes - 1],
                                                       list_of_words[num_for_indexes])
            tri_gram_last_element = tri_gram_prediction[0][-1]
        except ValueError:
            trigramError = True
        except IndexError:
            trigramError = True
        try:
            four_gram_prediction = my_fourgram.get_max_p(list_of_words[num_for_indexes - 2],
                                                         list_of_words[num_for_indexes - 1],
                                                         list_of_words[num_for_indexes])
            four_gram_last_element = four_gram_prediction[0][-1]
        except ValueError:
            fourgramError = True
        except IndexError:
            fourgramError = True
        final = ""
        # case 1
        if not fourgramError and not trigramError and not bigramError:
            print("bi-gram prediction: " + bi_gram_last_element)
            print("tri-gram prediction: " + tri_gram_last_element)
            print("four-gram prediction: " + four_gram_last_element)
            decision = input("please enter 1 for the bi-gram prediction, 2 for the tri-gram prediction or 3 for the "
                             "four-gram prediction\n")
            while decision not in ["1", "2", "3"]:
                print("invalid number")
                decision = input(
                    "please enter 1 for the bi-gram prediction, 2 for the tri-gram prediction or 3 for the "
                    "four-gram prediction\n")

            if decision == "1":
                list_of_words.append(bi_gram_last_element)
            elif decision == "2":
                list_of_words.append(tri_gram_last_element)
            elif decision == "3":
                list_of_words.append(four_gram_last_element)

            final = ' '.join(list_of_words)
            count = count + 1  # Print the final sentence
        # case 2
        elif not fourgramError and not trigramError and bigramError:
            print("tri-gram prediction: " + tri_gram_last_element)
            print("four-gram prediction: " + four_gram_last_element)
            decision = input("please enter 2 for the tri-gram prediction or 3 for the "
                             "four-gram prediction\n")
            while decision not in ["2", "3"]:
                print("invalid number")
                decision = input(
                    "please enter 2 for the tri-gram prediction or 3 for the "
                    "four-gram prediction\n")
            if decision == "2":
                list_of_words.append(tri_gram_last_element)
            elif decision == "3":
                list_of_words.append(four_gram_last_element)

            final = ' '.join(list_of_words)
            count = count + 1
        # case 3
        elif not fourgramError and trigramError and not bigramError:
            print("bi-gram prediction: " + bi_gram_last_element)
            print("four-gram prediction: " + four_gram_last_element)
            decision = input("please enter 1 for the bi-gram prediction or 3 for the "
                             "four-gram prediction\n")
            while decision not in ["1", "3"]:
                print("invalid number")
                decision = input(
                    "please enter 1 for the bi-gram prediction or 3 for the "
                    "four-gram prediction\n")
            if decision == "1":
                list_of_words.append(bi_gram_last_element)
            elif decision == "3":
                list_of_words.append(four_gram_last_element)

            final = ' '.join(list_of_words)
            count = count + 1
        # case 4
        elif fourgramError and not trigramError and not bigramError:
            print("bi-gram prediction: " + bi_gram_last_element)
            print("tri-gram prediction: " + tri_gram_last_element)
            decision = input("please enter 1 for the bi-gram prediction or 2 for the "
                             "tri-gram prediction\n")
            while decision not in ["1", "2"]:
                print("invalid number")
                decision = input(
                    "please enter 1 for the bi-gram prediction or 2 for the "
                    "tri-gram prediction\n")
            if decision == "1":
                list_of_words.append(bi_gram_last_element)
            elif decision == "2":
                list_of_words.append(tri_gram_last_element)

            final = ' '.join(list_of_words)
            count = count + 1
        # case 5
        elif fourgramError and trigramError and not bigramError:
            print("bi-gram prediction: " + bi_gram_last_element)
            decision = input("please enter 1 for the bi-gram prediction\n")
            while decision not in ["1"]:
                print("invalid number")
                decision = input("please enter 1 for the bi-gram prediction\n")
            if decision == "1":
                list_of_words.append(bi_gram_last_element)
            final = ' '.join(list_of_words)
            count = count + 1
        # case 6
        elif fourgramError and not trigramError and bigramError:
            print("tri-gram prediction: " + tri_gram_last_element)
            decision = input("please enter 2 for the tri-gram prediction\n")
            while decision not in ["2"]:
                print("invalid number")
                decision = input("please enter 2 for the tri-gram prediction\n")
            if decision == "2":
                list_of_words.append(tri_gram_last_element)
            final = ' '.join(list_of_words)
            count = count + 1
        # case 7
        elif not fourgramError and trigramError and bigramError:
            print("four-gram prediction: " + four_gram_last_element)
            decision = input("please enter 3 for the four-gram prediction\n")
            while decision not in ["3"]:
                print("invalid number")
                decision = input("please enter 3 for the four-gram prediction\n")
            if decision == "3":
                list_of_words.append(four_gram_last_element)
            final = ' '.join(list_of_words)
            count = count + 1
        # case 8
        else:
            print("Invalid sentence, please enter a new sentence or type 'quit' to end the program ")
        print(final)


if __name__ == "__main__":
    main()
