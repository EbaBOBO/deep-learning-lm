import tensorflow as tf
import numpy as np
from functools import reduce


def get_data(train_file, test_file):
    """
    Read and parse the train and test file line by line, then tokenize the sentences to build the train and test data separately.
    Create a vocabulary dictionary that maps all the unique tokens from your train and test data as keys to a unique integer value.
    Then vectorize your train and test data based on your vocabulary dictionary.

    :param train_file: Path to the training file.
    :param test_file: Path to the test file.
    :return: Tuple of train (1-d list or array with training words in vectorized/id form), test (1-d list or array with testing words in vectorized/id form), vocabulary (Dict containg index->word mapping)
    """
    # TODO: load and concatenate training data from training file.

    # TODO: load and concatenate testing data from testing file.

    # TODO: read in and tokenize training data

    # TODO: read in and tokenize testing data

    # BONUS: Ensure that all words appearing in test also appear in train

    # TODO: return tuple of training tokens, testing tokens, and the vocab dictionary.
    with open(train_file, "r") as f1:  
        data1 = f1.readlines() 
        vocab1 = set("".join(data1).split()) # collects all unique words in our dataset (vocab)
        word2id = {w: i for i, w in enumerate(list(vocab1))} # maps each word in our vocab to a unique index (label encode)
        # print(word2id)
        s1 = map(lambda x: x.split(), data1)
        train = []
        for sentence in s1:
            for word_index, word in enumerate(sentence):  
                train.append(word2id[word])
    # TODO: load and concatenate testing data from testing file.
    with open(test_file, "r") as f2:  
        data2 = f2.readlines() 
    # TODO: read in and tokenize training data

        s2 = map(lambda x: x.split(), data2)
        test = []
        for sentence in s2:
            for word_index, word in enumerate(sentence):  
                test.append(word2id[word])

    # TODO: read in and tokenize training data

    # TODO: read in and tokenize testing data

    # BONUS: Ensure that all words appearing in test also appear in train

    # TODO: return tuple of training tokens, testing tokens, and the vocab dictionary.
    train = np.array(train)
    test = np.array(test)

    return train,test,word2id
    pass
