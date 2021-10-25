import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from functools import reduce
from preprocess import get_data


class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class predicts the next words in a sequence.

        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()


        self.vocab_size = vocab_size
        self.embedding_size = 30 #TODO
        self.batch_size =  100 #TODO

        # TODO: initialize embeddings and forward pass weights (weights, biases)
        self.E = tf.Variable(tf.random.truncated_normal([self.vocab_size,self.embedding_size], stddev=0.1))
        # self.W = tf.Variable(tf.random.truncated_normal([self.embedding_size,self.vocab_size], stddev=0.1))
        self.W1 = tf.Variable(tf.random.truncated_normal([self.embedding_size*2,100], stddev=0.1))
        self.W2 = tf.Variable(tf.random.truncated_normal([100,1000], stddev=0.1))
        self.W3 = tf.Variable(tf.random.truncated_normal([1000,self.vocab_size], stddev=0.1))
        # self.b = tf.Variable(tf.random.truncated_normal([self.vocab_size], stddev=0.1))
        self.b1 = tf.Variable(tf.random.truncated_normal([100], stddev=0.1))
        self.b2 = tf.Variable(tf.random.truncated_normal([1000], stddev=0.1))
        self.b3 = tf.Variable(tf.random.truncated_normal([self.vocab_size], stddev=0.1))
    def call(self, inputs):
        """
        You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        :param inputs: word ids of shape (batch_size, 2)
        :return: probs: The batch element probabilities as a tensor of shape (batch_size, vocab_size)
        """

        #TODO: Fill in
        embedding = tf.nn.embedding_lookup(self.E, inputs, max_norm=None, name=None)
        embedding2 = tf.reshape(embedding,[self.batch_size,self.embedding_size*2])#[batchsize,60]

        logits1 = tf.add(tf.matmul(embedding2,self.W1),self.b1) # [batch_size,100]
        logits2 = tf.add(tf.matmul(logits1,self.W2),self.b2)  #[batch_size,1000]
        logits3 = tf.add(tf.matmul(logits2,self.W3),self.b3)    #[batch_size,vocab_size]
        return logits3
        pass

    def loss_function(self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction
        
        NOTE: Please use np.reduce_mean and not np.reduce_sum when calculating your loss.
        
        :param probs: a matrix of shape (batch_size, vocab_size)
        :return: the loss of the model as a tensor of size 1
        """
        #TODO: Fill in

        return tf.reduce_mean(tf.keras.metrics.sparse_categorical_crossentropy(labels, probs))
        # return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, probs))
        pass


def train(model, train_input, train_labels):
    """
    Runs through one epoch - all training examples. 
    You should take the train input and shape them into groups of two words.
    Remember to shuffle your inputs and labels - ensure that they are shuffled in the same order. 
    Also you should batch your input and labels here.
    :param model: the initilized model to use for forward and backward pass
    :param train_input: train inputs (all inputs for training) of shape (num_inputs,2)
    :param train_input: train labels (all labels for training) of shape (num_inputs,)
    :return: None
    """
    
    #TODO Fill in
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    seeds=[s for s in range(len(train_input))]
    # train_labels = np.array

    index=tf.random.shuffle(seeds)
    train_inputs1=tf.gather(train_input,index,axis=0)
    train_labels1=tf.gather(train_labels,index,axis=0)
    # train_inputs2=tf.image.random_flip_left_right(train_inputs1, 6).numpy()
    # loss1=[]
    for i in range(int(len(train_input)/model.batch_size)):
  # Implement backprop:
        with tf.GradientTape() as tape:
            logits=model.call(train_inputs1[i*model.batch_size:(i+1)*model.batch_size])
            losses=model.loss_function(logits,train_labels1[i*model.batch_size:(i+1)*model.batch_size])
            # loss1.append(losses)
      
        gradients = tape.gradient(losses, model.trainable_variables)
        # print(gradients)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    pass


def test(model, test_input, test_labels):
    """
    Runs through all test examples. You should take the test input and shape them into groups of two words.
    And test input should be batched here as well.
    :param model: the trained model to use for prediction
    :param test_input: train inputs (all inputs for testing) of shape (num_inputs,2)
    :param test_input: train labels (all labels for testing) of shape (num_inputs,)
    :returns: perplexity of the test set
    """
    
    #TODO: Fill in
    #NOTE: Ensure a correct perplexity formula (different from raw loss)
    sum = 0
    for i in range(int(len(test_input)/model.batch_size)):
# Implement backprop:
        with tf.GradientTape() as tape:
            logits=model.call(test_input[i*model.batch_size:(i+1)*model.batch_size])
            losses=model.loss_function(logits,test_labels[i*model.batch_size:(i+1)*model.batch_size])
            sum += losses
        
    perplexity = tf.exp(sum/len(test_input))
        # print(gradients)

    return perplexity
    pass  


def generate_sentence(word1, word2, length, vocab, model):
    """
    Given initial 2 words, print out predicted sentence of targeted length.

    :param word1: string, first word
    :param word2: string, second word
    :param length: int, desired sentence length
    :param vocab: dictionary, word to id mapping
    :param model: trained trigram model

    """

    #NOTE: This is a deterministic, argmax sentence generation

    reverse_vocab = {idx: word for word, idx in vocab.items()}
    output_string = np.zeros((1, length), dtype=np.int)
    output_string[:, :2] = vocab[word1], vocab[word2]

    for end in range(2, length):
        start = end - 2
        output_string[:, end] = np.argmax(
            model(output_string[:, start:end]), axis=1)
    text = [reverse_vocab[i] for i in list(output_string[0])]

    print(" ".join(text))


def main():
    train1,test1,word2id = get_data('/Users/zccc/1470projects/data/train.txt','/Users/zccc/1470projects/data/test.txt')
    # print(train1.shape) #(1465614,)
    # print(test1.shape) #(361912,)

    # TO-DO:  Separate your train and test data into inputs and labels
    train_input = np.zeros((int(len(train1)/3),2),dtype=np.int32)
    train_labels = np.zeros((int(len(train1)/3),),dtype=np.int32)
    for i in range(int(len(train1)/3)):
        train_input[i] = train1[i*3:i*3+2]
        train_labels[i] = train1[i*3+2]
    # print(train_input.shape) #(488538, 2)

    test_input = np.zeros((int(len(test1)/3),2))
    test_labels = np.zeros((int(len(test1)/3),))
    for i in range(int(len(test1)/3)):
        test_input[i] = test1[i*3:i*3+2]
        test_labels[i] = test1[i*3+2]
    # print(test_input.shape) #(120637, 2)
    # TODO: initialize model
    obj = Model(len(word2id))

    # TODO: Set-up the training step
    train(obj,train_input,train_labels)
    # TODO: Set up the testing steps
    perplexity = test(obj,test_input,test_labels)
    print('perplexity is: ',perplexity)
    # Print out perplexity 
    
    # BONUS: Try printing out sentences with different starting words   
    pass

if __name__ == '__main__':
    main()
