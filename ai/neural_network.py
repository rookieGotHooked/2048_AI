# coding: utf-8
import os
import numpy as np
from params import Directions, TARGET
from game.game import Game


class NeuralNetwork:
    """
    Represents a neural network
    """

    def __init__(self, function="tanh", alpha=0.01):
        self._layers = []
        self.function = function
        self.alpha = alpha

        if function == "sigmoid":
            self.NORMALIZED_DIR_DICT = {'Up': 1.0, 'Down': 0.75, 'Left': 0.5, 'Right': 0.25}
            self.DIRECTION_VALUES_LIST = [1.0, 0.75, 0.5, 0.25]
            self.DIRECTIONS_LIST = [Directions.Up, Directions.Down, Directions.Left, Directions.Right]

        elif function == "tanh":
            self.NORMALIZED_DIR_DICT = {'Up': -1.0, 'Down': -0.5, 'Left': 0.5, 'Right': 1.0}
            self.DIRECTION_VALUES_LIST = [-1.0, -0.5, 0.5, 1.0]
            self.DIRECTIONS_LIST = [Directions.Up, Directions.Down, Directions.Left, Directions.Right]

        elif function == "leaky_relu" or "relu":
            self.NORMALIZED_DIR_DICT = {'Up': 10.0, 'Down': 7.5, 'Left': 5.0, 'Right': 2.5}
            self.DIRECTION_VALUES_LIST = [10.0, 7.5, 5.0, 2.5]
            self.DIRECTIONS_LIST = [Directions.Up, Directions.Down, Directions.Left, Directions.Right]

    def add_layer(self, layer):
        """
        Adds a layer to the neural network

        @param layer: The layer to add
        @type layer: Layer
        """
        self._layers.append(layer)

    def feed_forward(self, x):
        """
        Feed forward the input through the layers

        @param x: The input values
        @type x: np.array

        @return: The result
        @rtype X: np.array
        """
        for layer in self._layers:
            if self.function == "leaky_relu" or "relu":
                x = layer.activate(x, self.function, self.alpha)
            else:
                x = layer.activate(x, self.function)
        # print(f"feed_forward x: {x}")
        return x

    def predict(self, x):
        """
        Function to predict the next direction to play given the current Grid (x)

        @param x: The input values
        @type x: np.array

        @return: The ordered list of directions to play (0: first choice, 1: second choice, etc.)
        @rtype: list of Constants.Directions
        """
        ff = self.feed_forward(x)
        output_error_vector = np.power(np.subtract(np.array(self.DIRECTION_VALUES_LIST), ff), 2)
        direction_vector_choices = np.argsort(output_error_vector)
        choices_to_return = list()
        for index in np.nditer(direction_vector_choices):
            choices_to_return.append(self.DIRECTIONS_LIST[int(index)])

        # print('Result after predictions:')
        # for choice in choices_to_return:
        #     print(choice.value)

        return choices_to_return

    def backpropagation(self, x, y, learning_rate):
        """
        Performs the backward propagation algorithm and updates the layers weights
        Source: https://blog.zhaytam.com/2018/08/15/implement-neural-network-backpropagation/

        @param x: The input values
        @type x: np.array
        @param y: The target values
        @type y: np.array
        @param learning_rate: The learning rate (between 0 and 1)
        @type learning_rate: float
        """
        # Feed forward for the output
        output = self.feed_forward(x)

        # Loop over the layers backward
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]

            # If this is the output layer
            if layer == self._layers[-1]:
                layer.error = y - output
                # The output = layer.last_activation in this case
                layer.delta = layer.error * layer.apply_activation_derivative(output, self.function)
            else:
                next_layer = self._layers[i + 1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * layer.apply_activation_derivative(layer.last_activation, self.function)

        # Update the weights
        for i in range(len(self._layers)):
            layer = self._layers[i]
            # The input is either the previous layers output or X itself (for the first hidden layer)
            input_to_use = np.atleast_2d(x if i == 0 else self._layers[i - 1].last_activation)
            layer.weights += layer.delta * input_to_use.T * learning_rate

    def train(self, x, y, learning_rate, max_epochs):
        """
        Trains the neural network using backpropagation
        Source: https://blog.zhaytam.com/2018/08/15/implement-neural-network-backpropagation/

        @param x: The input values
        @type x: np.array
        @param y: The target values
        @type y: np.array
        @param learning_rate: The learning rate (between 0 and 1)
        @type learning_rate: float
        @param max_epochs: The maximum number of epochs (cycles)
        @type max_epochs: int

        @return: The list of calculated MSE errors
        @rtype: list(float)
        """
        mses = []
        if x is not None:  # If some train data is available
            for i in range(max_epochs):
                for j in range(len(x)):
                    self.backpropagation(x[j], y[j], learning_rate)
                if i % 10 == 0:
                    mse = np.mean(np.square(y - self.feed_forward(x)))
                    mses.append(mse)
                    print('Epoch: #%s, MSE: %f' % (i, float(mse)))
        return mses

    def train_from_directory(self, directory, learning_rate, max_epochs):
        """
        Train a neural network based on a set of game logs located in a single directory

        @param directory: the path to the directory containing the 2048 log files
        @type directory: str
        @param learning_rate: The learning rate (between 0 and 1)
        @type learning_rate: float
        @param max_epochs: The maximum number of epochs (cycles)
        @type max_epochs: int
        """
        all_x = None
        all_y = None
        for filename in os.listdir(directory):
            if filename.endswith(".log"):
                print("OK - File parsed: {}".format(os.path.join(directory, filename)))
                game_obj = Game()
                game = game_obj.load_game(log_file_path=os.path.join(directory, filename), display_grid=False)
                x, y = self.parse_inputs_outputs_for_neural_net(game)
                if all_x is None:
                    all_x = x
                    all_y = y
                else:
                    all_x = np.concatenate((all_x, x))
                    all_y = np.concatenate((all_y, y))
            else:
                print("NOK - File not parsed: {}".format(os.path.join(directory, filename)))
        self.train(all_x, all_y, learning_rate, max_epochs)

    # @staticmethod
    def parse_inputs_outputs_for_neural_net(self, game):
        """
        Helper methode to convert a 2048 log file into a list of (x,y) to train the neural network

        @param game: An existing 2048 game
        @type game: Game

        @return: Inputs/Outputs for every history step
        @rtype: tuple of (np.array, np.array)
        """
        nb_training_examples = len(game.history.grid_history)
        x = np.zeros((nb_training_examples, 4 * 4))
        y = np.zeros((nb_training_examples, 1))
        for i in range(len(game.history.grid_history)):
            if game.history.direction_state_history[i] in ['Up', 'Down', 'Left', 'Right']:
                x[i, :] = [float(num) / TARGET for num in game.history.grid_history[i].strip().split(' ')]
                y[i, :] = [self.NORMALIZED_DIR_DICT[game.history.direction_state_history[i]]]
        return x, y
