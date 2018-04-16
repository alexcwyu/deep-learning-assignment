from keras import layers, models, optimizers, regularizers, initializers
from keras import backend as K
from .model_helper import build_dense_layer

class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high,
                 batch_normalized, dropout, dropout_rate, learning_rate, beta1):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        self.batch_normalized = batch_normalized
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.beta1 = beta1

        # Initialize any other variables here

        self.build_model()



    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        kernel_regularizer= regularizers.l2(0.01)
        #kernel_regularizer= None
        kernel_initializer = initializers.RandomNormal()
        #kernel_initializer= None
        layer1 = build_dense_layer(states, units=32, activation='relu', kernel_regularizer= kernel_regularizer, kernel_initializer = kernel_initializer, batch_normalized = self.batch_normalized, dropout = self.dropout, dropout_rate = self.dropout_rate)

        layer2 = build_dense_layer(layer1, units=64, activation='relu', kernel_regularizer= kernel_regularizer, kernel_initializer = kernel_initializer, batch_normalized = self.batch_normalized, dropout = self.dropout, dropout_rate = self.dropout_rate)

        layer3 = build_dense_layer(layer2, units=32, activation='relu', kernel_regularizer= kernel_regularizer, kernel_initializer = kernel_initializer, batch_normalized = self.batch_normalized, dropout = self.dropout, dropout_rate = self.dropout_rate)


        # Add final output layer with sigmoid activation
        raw_actions = build_dense_layer(layer3, units=self.action_size, activation='sigmoid', kernel_regularizer = kernel_regularizer, kernel_initializer = kernel_initializer, batch_normalized = False, dropout = False, name='raw_actions')

        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
                                name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr = self.learning_rate, beta_1=self.beta1)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)
