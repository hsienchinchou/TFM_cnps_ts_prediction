
"""

Aggregator(s) for the NP

"""

import tensorflow as tf


class MeanAggregator:

    def __init__(self, name="MeanAggregator", reuse=None):
        self._name = name
        self._reuse = reuse

    def __call__(self, representation_list):
        """
        Construct the graph.
        :param representation_list: A matrix of representation vectors (shape [batch_size, num_context, rep_vec_size])
        :return: [batch_size, rep_vec_size] tensor, averaged over all context points.
        """
        with tf.variable_scope(self._name, reuse=self._reuse):
            return tf.reduce_mean(representation_list, 1)

class RNNAggregator:

    def __init__(self, name="RNNAggregator", reuse=None):
        self._name = name
        self._reuse = reuse
        self.model = tf.keras.Sequential()
        

    def __call__(self, representation_list):
        """
        Construct the graph.
        :param representation_list: A matrix of representation vectors (shape [batch_size, num_context, rep_vec_size])
        :return: [batch_size, rep_vec_size] tensor
        """
        with tf.variable_scope(self._name, reuse=self._reuse):
            _batch_size, _, rep_vec_size = representation_list.shape.as_list()
            
            # Add an input layer
            self.model.add(tf.keras.layers.InputLayer(input_tensor=representation_list))
            
            # Add a LSTM layer with 5 internal units.
            self.model.add(tf.keras.layers.LSTM(5, recurrent_activation='relu', activation='tanh', kernel_initializer='lecun_uniform'))
            
            # Add a output layer with 'rep_vec_size' units.
            self.model.add(tf.keras.layers.Dense(rep_vec_size, activation='tanh'))
            
            return tf.reshape(self.model.output, (_batch_size, rep_vec_size))