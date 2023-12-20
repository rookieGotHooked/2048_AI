import tkinter as tk
from params import (GAME_VIEW_WIDTH, WINDOW_HEIGHT,
                    TK_FONT, TK_FONT_SIZE, FPS, Directions, DATA_DIR_NAME, TRAIN_DIR_NAME, NEURAL_NET_TRAINING_RATE,
                    NEURAL_NET_MAX_EPOCHS)
from game.game import Game

import numpy as np
from ai.neural_network import NeuralNetwork
from ai.layer import Layer
from os import path

class Window:
    def __init__(self):
        self.window = tk.Tk()
        self.window.geometry(f"{GAME_VIEW_WIDTH}x{WINDOW_HEIGHT}")

        self.gameView = tk.Label(self.window)
        self.gameView.place(x=-1, y=-1)
        self.window_after_id = None

        self.game = Game()
        # self.window.bind('<KeyPress>', self.handle_key_press)

        self.score_label = tk.Label(self.window, text=f"Score: {self.game.score}", font=(TK_FONT, TK_FONT_SIZE, "bold"), padx=10)
        self.score_label.place(x=0, y=420)

        self.result_label = tk.Label(self.window, text=f"Game Over...",
                                font=(TK_FONT, TK_FONT_SIZE, "bold"), padx=10)
        self.result_label.place(x=-100, y=-100)

        self.decision = tk.Label(self.window, text=f"Next move: ", font=(TK_FONT, TK_FONT_SIZE, "bold"), padx=10)
        self.decision.place(x=0, y=460)

        self.nn = NeuralNetwork()
        self.nn.add_layer(Layer(16, 4))
        self.nn.add_layer(Layer(4, 4))

        train_dir = path.join(DATA_DIR_NAME, TRAIN_DIR_NAME)
        self.nn.train_from_directory(directory=train_dir,
                                     learning_rate=NEURAL_NET_TRAINING_RATE,
                                     max_epochs=NEURAL_NET_MAX_EPOCHS)

        self.replay_dir = path.join(DATA_DIR_NAME,
                                    'replays',
                                    '{}_{}'.format(4, 4))

        self.update()

        self.window.mainloop()

    def update(self):
        image = self.game.update()
        self.gameView.image = image
        self.gameView.config(image=image)

        self.score_label.config(text=f"Score: {self.game.score}")

        self.window_after_id = self.window.after(int(1000 / FPS), self.update)

        if not self.game.game_end:
            x = np.array(np.mat(self.game.to_string()))
            predictions = self.nn.predict(x)
            self.game.play_many_directions(predictions)
            self.game.save_game(base_path=self.replay_dir)
        elif self.game.game_end:
            self.result_label.place(x=125, y=150)
            self.stop_game()


    # def handle_key_press(self, event):
    #     if event.keysym == 'Up':
    #         self.game.direction = Directions.Up
    #     elif event.keysym == 'Down':
    #         self.game.direction = Directions.Down
    #     elif event.keysym == 'Left':
    #         self.game.direction = Directions.Left
    #     elif event.keysym == 'Right':
    #         self.game.direction = Directions.Right
    #
    #     self.game.move(self.game.direction, self.game.current_board)

    def stop_game(self):
        self.game.gameMain.quit()
        self.window.after_cancel(self.window_after_id)
