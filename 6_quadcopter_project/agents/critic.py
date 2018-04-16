from keras import layers, models, optimizers, regularizers, initializers
from keras import backend as K
from .model_helper import build_dense_layer


class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size,
                 batch_normalized, dropout, dropout_rate, learning_rate, beta1):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.batch_normalized = batch_normalized
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.beta1 = beta1

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway

        kernel_regularizer= regularizers.l2(0.01)
        #kernel_regularizer= None

        kernel_initializer = initializers.RandomNormal()
        #kernel_initializer = None

        net_states1 = build_dense_layer(states, units=32, activation='relu', kernel_regularizer= kernel_regularizer, kernel_initializer = kernel_initializer, batch_normalized = self.batch_normalized, dropout = self.dropout, dropout_rate = self.dropout_rate)
        net_states2 = build_dense_layer(net_states1, units=64, activation='relu', kernel_regularizer= kernel_regularizer, kernel_initializer = kernel_initializer, batch_normalized = self.batch_normalized, dropout = self.dropout, dropout_rate = self.dropout_rate)


        # Add hidden layer(s) for action pathway
        net_actions1 = build_dense_layer(actions, units=32, activation='relu', kernel_regularizer= kernel_regularizer, kernel_initializer = kernel_initializer, batch_normalized = self.batch_normalized, dropout = self.dropout, dropout_rate = self.dropout_rate)
        net_actions2 = build_dense_layer(net_actions1, units=64, activation='relu', kernel_regularizer= kernel_regularizer, kernel_initializer = kernel_initializer, batch_normalized = self.batch_normalized, dropout = self.dropout, dropout_rate = self.dropout_rate)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = layers.Add()([net_states2, net_actions2])
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed

        # Add final output layer to prduce action values (Q values)

        Q_values = build_dense_layer(net, units=1, activation=None, kernel_regularizer= kernel_regularizer, kernel_initializer = kernel_initializer, batch_normalized = False, dropout = False, name='q_values')

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr = self.learning_rate, beta_1=self.beta1)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)
