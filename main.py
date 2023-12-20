from ui.window import Window
from ai.neural_network import NeuralNetwork
from ai.layer import Layer
from os import path
import params
import numpy as np

if __name__ == '__main__':
    # nn = NeuralNetwork()
    # nn.add_layer(Layer(16, 4))
    # nn.add_layer(Layer(4, 4))
    #
    # train_dir = path.join(params.DATA_DIR_NAME, params.TRAIN_DIR_NAME)
    # nn.train_from_directory(directory=train_dir,
    #                         learning_rate=params.NEURAL_NET_TRAINING_RATE,
    #                         max_epochs=params.NEURAL_NET_MAX_EPOCHS)

    game_window = Window()
    # current_game = game_window.game
    #
    # replay_dir = path.join(params.DATA_DIR_NAME,
    #                        'replays',
    #                        '{}_{}'.format(4, 4))
    #
    # while not game_window.game.game_end:
    #     x = np.array(np.mat(current_game.to_string()))
    #     predictions = nn.predict(x)
    #     current_game.play_many_directions(predictions)
    # current_game.save_game(base_path=replay_dir)
