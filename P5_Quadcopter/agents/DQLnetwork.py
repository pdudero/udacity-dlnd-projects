import tensorflow as tf

class QNetwork:
    def __init__(self, state_size, action_size, learning_rate=0.01, name='QNetwork'):
        # state inputs to the Q-network
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size], name='inputs')
            
            # One hot encode the actions to later choose the Q-value for the action
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')  # actions_ must be (batch_size)
            one_hot_actions = tf.one_hot(self.actions_, action_size)          # (batch_size,action_size)
            
            # Target Q values for training
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')
            
            # ReLU hidden layers
            self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, 128)
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, 256)
            self.fc3 = tf.contrib.layers.fully_connected(self.fc2, 128)

            # Linear output layer
            self.output = tf.contrib.layers.fully_connected(self.fc3, 
                                                            action_size, 
                                                            activation_fn=None)

            ### Train with loss (targetQ - Q)^2
            # This next line chooses feature_size values from output (per batch row)
            #   according to the one-hot encoded actions.
            self.Qs = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1) # (batch_size)
            
            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Qs))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)