import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from preprocess import get_data


class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class predicts the next words in a sequence.

        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        # TODO: initialize vocab_size, embedding_size

        self.vocab_size = vocab_size
        self.window_size = 20 # DO NOT CHANGE!
        self.embedding_size = 30 #TODO
        self.batch_size = 100 #TODO 

        # TODO: initialize embeddings and forward pass weights (weights, biases)
        # Note: You can now use tf.keras.layers!
        # - use tf.keras.layers.Dense for feed forward layers: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
        # - and use tf.keras.layers.GRU or tf.keras.layers.LSTM for your RNN 
        self.E = tf.Variable(tf.random.truncated_normal([self.vocab_size,self.embedding_size], stddev=0.1))
        self.LSTM = tf.keras.layers.LSTM(100,return_sequences=True, return_state=True)
        self.dense_layer1 = tf.keras.layers.Dense(100)
        self.dense_layer2 = tf.keras.layers.Dense(1000)
        self.dense_layer3 = tf.keras.layers.Dense(self.vocab_size)
    def call(self, inputs, initial_state):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.

        :param inputs: word ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch element probabilities as a tensor, a final_state (Note 1: If you use an LSTM, the final_state will be the last two RNN outputs, 
        Note 2: We only need to use the initial state during generation)
        using LSTM and only the probabilites as a tensor and a final_state as a tensor when using GRU 
        """
        
        #TODO: Fill in 
        # print(inputs.shape)
        embedding = tf.nn.embedding_lookup(self.E, inputs, max_norm=None, name=None)
        output1, final_memory_state, final_carry_state = self.LSTM(embedding)
        logits1 = self.dense_layer1(output1)
        logits2 = self.dense_layer2(logits1)
        logits3 = self.dense_layer3(logits2)
        probs = tf.nn.softmax(logits3)
        # print('probs',probs.shape)

        return probs,final_memory_state

    def loss(self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction
        
        NOTE: You have to use np.reduce_mean and not np.reduce_sum when calculating your loss

        :param logits: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """

        #TODO: Fill in
        #We recommend using tf.keras.losses.sparse_categorical_crossentropy
        #https://www.tensorflow.org/api_docs/python/tf/keras/losses/sparse_categorical_crossentropy
        # print('labels:',labels.shape)
        # print(probs.shape)
    
        return tf.reduce_mean(tf.keras.metrics.sparse_categorical_crossentropy(labels, probs))

        # return None


def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
#     #TODO: Fill in
    print('train')
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    train_inputs1 = np.zeros((int(len(train_inputs)/model.window_size)-1,model.window_size),dtype=np.int32)
    train_labels1 = np.zeros((int(len(train_labels)/model.window_size)-1,model.window_size),dtype=np.int32)
    # print(train_inputs1.shape)
    # print('train_inputs',train_inputs.shape)
    for i in range(int(len(train_inputs)/model.window_size)-1):
        train_inputs1[i] = train_inputs[i*model.window_size:i*model.window_size+model.window_size]
        train_labels1[i] = train_inputs[i*model.window_size+1:i*model.window_size+model.window_size+1]
    # train_inputs1 = np.reshape(train_inputs1,-1)
    # train_labels1 = np.reshape(train_labels1,-1)

    seeds=[s for s in range(len(train_labels1))]
    index=tf.random.shuffle(seeds)
    train_inputs2=tf.gather(train_inputs1,index,axis=0)
    train_labels2=tf.gather(train_labels1,index,axis=0)

    for i in range(int(len(train_inputs2)/model.batch_size)):
  # Implement backprop:
        with tf.GradientTape() as tape:
            probs,final_memory_state=model.call(train_inputs2[i*model.batch_size:(i+1)*model.batch_size],None)
            # print('train probs',probs.shape)
            losses=model.loss(probs,train_labels2[i*model.batch_size:(i+1)*model.batch_size])

 
        gradients = tape.gradient(losses, model.trainable_variables)
        # print(gradients)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    pass


def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set
    """
    
    #TODO: Fill in
    #NOTE: Ensure a correct perplexity formula (different from raw loss)
    test_inputs1 = np.zeros((int(len(test_inputs)/model.window_size)-1,model.window_size),dtype=np.int32)
    test_labels1 = np.zeros((int(len(test_labels)/model.window_size)-1,model.window_size),dtype=np.int32)
    for i in range(int(len(test_inputs)/model.window_size)-1):
        test_inputs1[i] = test_inputs[i*model.window_size:i*model.window_size+model.window_size]
        test_labels1[i] = test_inputs[i*model.window_size+1:i*model.window_size+model.window_size+1]
        # print(test_labels1[i].shape)
    sum = 0
    for i in range(int(len(test_inputs1)/model.batch_size)):
        # print(i)
        with tf.GradientTape() as tape:
            logits,final_memory_state = model.call(test_inputs1[i*model.batch_size:(i+1)*model.batch_size],None)
            # print('logits ',logits) #(100,20,4962)
            losses=model.loss(logits,test_labels1[i*model.batch_size:(i+1)*model.batch_size])
            # print('losses:',losses)
            sum += losses
        
    perplexity = np.exp(sum/int(len(test_inputs1)/model.batch_size))
    # print('perplexity is:',perplexity)

    return perplexity
    pass  


def generate_sentence(word1, length, vocab, model, sample_n=10):
    """
    Takes a model, vocab, selects from the most likely next word from the model's distribution

    :param model: trained RNN model
    :param vocab: dictionary, word to id mapping
    :return: None
    """

    #NOTE: Feel free to play around with different sample_n values

    reverse_vocab = {idx: word for word, idx in vocab.items()}
    previous_state = None

    first_string = word1
    first_word_index = vocab[word1]
    next_input = [[first_word_index]]
    text = [first_string]

    for i in range(length):
        logits, previous_state = model.call(next_input, previous_state)
        logits = np.array(logits[0,0,:])
        top_n = np.argsort(logits)[-sample_n:]
        n_logits = np.exp(logits[top_n])/np.exp(logits[top_n]).sum()
        out_index = np.random.choice(top_n,p=n_logits)

        text.append(reverse_vocab[out_index])
        next_input = [[out_index]]

    print(" ".join(text))


def main():
    # TO-DO: Pre-process and vectorize the data
    # HINT: Please note that you are predicting the next word at each timestep, so you want to remove the last element
    # from train_x and test_x. You also need to drop the first element from train_y and test_y.
    # If you don't do this, you will see impossibly small perplexities.
    
    # TO-DO:  Separate your train and test data into inputs and labels

    train1,test1,word2id = get_data('/Users/zccc/1470projects/data/train.txt','/Users/zccc/1470projects/data/test.txt')
    # TODO: initialize model
    obj = Model(len(word2id))

    # TODO: Set-up the training step
    # train(obj,train1,train1)
    # TODO: Set up the testing steps
    perplexity = test(obj,test1,test1)
    print('perplexity is: ',perplexity)
    pass

if __name__ == '__main__':
    main()
