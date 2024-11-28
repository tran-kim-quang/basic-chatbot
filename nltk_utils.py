import numpy as np
import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    #split sentence into array of word
    return nltk.word_tokenize(sentence)

def stem(word):
    #find the root of the word
    return stemmer.stem(word.lower())

def bag_word(tokenize_sentence, words):
    #if a known word that exists in sentence, 0 for otherwise
    """
    ex: 
    sentence = "hi there, how's goin!"
    words = ['hi', 'hello','there', 'what', 'goin']
    bag   = [  1,       0,      1,      0,      1]
    """ 
    sentence_words = [stem(word) for word in tokenize_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for index, w in enumerate(words):
        if w in sentence_words:
            bag[index] = 1
    return bag