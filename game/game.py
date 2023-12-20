import time
from os import path

import numpy as np
import re
import pygame
from params import (GAME_VIEW_WIDTH, GAME_VIEW_HEIGHT, COLORS, FONT, BOARD_INIT_VALUES, Directions, States, TARGET)
from PIL import ImageTk, Image
from game.history import History
import random
from pathlib import Path


class Game:

    def __init__(self, play=False):
        self.gameMain = pygame
        self.game_end = False
        self.score = 0
        self.round_count = 0
        self.gameMain.init()
        self.surface = self.gameMain.Surface([GAME_VIEW_WIDTH, GAME_VIEW_HEIGHT])
        self.full = False
        self.history = History(4, 4)
        self.current_board = BOARD_INIT_VALUES
        self.current_board, self.game_end = self.new_piece(self.current_board)
        self.current_board, self.game_end = self.new_piece(self.current_board)
        self.draw_board()
        self.draw_pieces(self.current_board)
        self.direction = ''
        if play:
            self.history.add_grid_state(self.to_string(), self.score)

        #
        # while not game_window.game.game_end:
        #     x = np.array(np.mat(current_game.to_string()))
        #     predictions = nn.predict(x)
        #     current_game.play_many_directions(predictions)
        # current_game.save_game(base_path=replay_dir)

    def update(self):
        self.draw_board()
        self.draw_pieces(self.current_board)
        image = self.gameMain.surfarray.array3d(self.surface)

        np_image = np.array(image)
        np_image = np.flip(np_image, axis=1)
        np_image = np.rot90(np_image, k=1)

        out_image = Image.fromarray(np_image)

        return ImageTk.PhotoImage(out_image)

    def draw_board(self):
        pygame.draw.rect(self.surface, COLORS['bg'], [0, 0, 400, 400], 0, 0)

    def new_piece(self, board):
        full = False
        game_end = False
        while any(0 in row for row in board):
            row = random.randint(0, 3)
            col = random.randint(0, 3)
            if board[row][col] == 0:
                if random.randint(1, 10) == 10:
                    board[row][col] = 4
                    return board, full
                else:
                    board[row][col] = 2
                    return board, full

        full = True
        if full:
            for row in range(3):
                for col in range(3):
                    if (col < len(board[row]) - 1 and board[row][col + 1] == board[row][col]) \
                            or (row < len(board) - 1 and board[row + 1][col] == board[row][col]):
                        return board, game_end

        game_end = True

        return board, game_end

    def is_winning(self):
        return np.count_nonzero(self.current_board == TARGET) > 0

    def draw_pieces(self, board):
        for i in range(4):
            for j in range(4):
                value = board[i][j]
                if value > 8:
                    value_color = COLORS['light_text']
                else:
                    value_color = COLORS['dark_text']

                if value <= 2048:
                    color = COLORS[value]
                else:
                    color = COLORS['other']

                pygame.draw.rect(self.surface, color, [j * 95 + 20, i * 95 + 20, 75, 75])

                if value > 0:
                    value_len = len(str(value))
                    font = self.gameMain.font.SysFont(FONT, 48 - (5 * value_len))
                    value_text = font.render(str(value), True, value_color)
                    text_rect = value_text.get_rect(center=(j * 95 + 57, i * 95 + 57))
                    self.surface.blit(value_text, text_rect)
                    pygame.draw.rect(self.surface, 'black', [j * 95 + 20, i * 95 + 20, 75, 75], 2)

    def move(self, direction, board):
        should_spawn = False
        merged = [[False for _ in range(4)] for _ in range(4)]

        if direction == Directions.Up:
            for row in range(4):
                for col in range(4):
                    shift = 0
                    if row > 0:
                        for r in range(row):
                            if board[r][col] == 0:
                                shift += 1
                        if shift > 0 and board[row][col] != 0:
                            board[row - shift][col] = board[row][col]
                            board[row][col] = 0
                            should_spawn = True
                        if board[row - shift - 1][col] == board[row - shift][col] and not merged[row - shift][
                            col] and not merged[row - shift - 1][col]:
                            board[row - shift - 1][col] *= 2
                            self.score += board[row - shift - 1][col]
                            board[row - shift][col] = 0
                            merged[row - shift - 1][col] = True

        elif direction == Directions.Down:
            for row in range(3):
                for col in range(4):
                    shift = 0
                    for r in range(row + 1):
                        if board[3 - r][col] == 0:
                            shift += 1

                    if shift > 0 and board[2 - row][col] != 0:
                        board[2 - row + shift][col] = board[2 - row][col]
                        board[2 - row][col] = 0
                        should_spawn = True
                    if 3 - row + shift <= 3:
                        if board[2 - row + shift][col] == board[3 - row + shift][col] and not merged[3 - row + shift][
                            col] and not merged[2 - row + shift][col]:
                            board[3 - row + shift][col] *= 2
                            self.score += board[3 - row + shift][col]
                            board[2 - row + shift][col] = 0
                            merged[3 - row + shift][col] = True

        elif direction == Directions.Left:
            for row in range(4):
                for col in range(4):
                    shift = 0
                    if col > 0:
                        for c in range(col):
                            if board[row][c] == 0:
                                shift += 1

                        if shift > 0 and board[row][col] != 0:
                            board[row][col - shift] = board[row][col]
                            board[row][col] = 0
                            should_spawn = True
                        if board[row][col - shift] == board[row][col - shift - 1] and not merged[row][
                            col - shift - 1] and not merged[row][col - shift]:
                            board[row][col - shift - 1] *= 2
                            self.score += board[row][col - shift - 1]
                            board[row][col - shift] = 0
                            merged[row][col - shift - 1] = True

        elif direction == Directions.Right:
            for row in range(4):
                for col in range(4):
                    shift = 0
                    for c in range(col):
                        if board[row][3 - c] == 0:
                            shift += 1
                    if shift > 0 and board[row][3 - col] != 0:
                        board[row][3 - col + shift] = board[row][3 - col]
                        board[row][3 - col] = 0
                        should_spawn = True
                    if 4 - col + shift <= 3:
                        if board[row][4 - col + shift] == board[row][3 - col + shift] and not merged[row][
                            4 - col + shift] and not merged[row][3 - col + shift]:
                            board[row][4 - col + shift] *= 2
                            self.score += board[row][4 - col + shift]
                            board[row][3 - col + shift] = 0
                            merged[row][4 - col + shift] = True

        if should_spawn or any(True in row for row in merged):
            self.current_board, self.game_end = self.new_piece(self.current_board)

        return board, should_spawn

    def load_game(self, log_file_path, display_grid=True):
        """
        Static method to load a 2048 game log to visualize it through the GUI

        @param log_file_path: the path of the log to load
        @type log_file_path: str
        @param display_grid: whether or not to print the initial state of the 2048 game
        @type display_grid: bool

        @return: a Game object that contains all the game history (tile positions, directions and score)
        @rtype: Game
        """
        with open(log_file_path, 'r') as f:
            line = f.readline()
            while line:
                l_split = line.strip().split(' ')
                if len(l_split) > 3:
                    self.history.score_history.append(int(l_split[1]))
                    self.history.grid_history.append(' '.join(l_split[2:-2]))
                    if l_split[-2] != "WIN" and l_split[-2] != "LOOSE":
                        self.history.direction_state_history.append(Directions(l_split[-2]))
                    else:
                        self.history.direction_state_history.append(States(l_split[-2]))
                    m = re.search(r"\[([-0-9]+)\]", l_split[-1])  # Regex to extract index from brackets
                    self.history.direction_index_history.append(int(m.group(1)))
                line = f.readline()
        return self

    def to_string(self):
        str_to_return = ""
        for r in range(4):
            for c in range(4):
                str_to_return += "{} ".format(self.current_board[r][c])
        return str_to_return.strip()

    def play_one_direction(self, direction, index_choice):
        """
        Method to play one direction on the Grid object associated to this Game

        @param direction: one of the defined directions from the Constants file
        @type direction: Constants.Directions
        @param index_choice: the index of the chosen direction (0 was the first choice, 3 was the last choice)
        @type index_choice: int

        @return: whether or not it was a valid move (i.e. at least one tile moved)
        @rtype: bool
        """
        # self.grid.move_tiles(direction)
        self.move(direction, self.current_board)
        # self.score += self.grid.merge(direction)
        # self.grid.move_tiles(direction)
        # self.move(direction, self.current_board)
        if self.history.something_moved(self.to_string()):
            self.history.add_direction_or_state(direction, index_choice)
            print("Next direction to be played: {}".format(direction.value))
            print("=======================================")  # To distinguish from next round
            self.round_count += 1
            # self.grid.generate_new_number(self.grid.return_free_positions())
            self.history.add_grid_state(self.to_string(), self.score)
            print(self.__repr__())
            return True
        else:
            return False

    def play_many_directions(self, direction_list):
        """
        Method to play one direction from a list on the Grid object associated to this Game
        The method plays one direction at maximum and then exits if it was a valid move
        If no direction can be played, the method exits

        @param direction_list: the directions to be played sorted by order of preference (index 0 will be tried first)
        @type direction_list: list of Constants.Directions
        """
        for i in range(len(direction_list)):
            if self.play_one_direction(direction_list[i], i):
                break

    def save_game(self, base_path):
        """
        Utility method to save the History of a Game into file for later inspection

        @param base_path: the complete directory path where to write the log file
        @type base_path: str
        """
        Path(base_path).mkdir(parents=True, exist_ok=True)  # Require Python 3.4+
        file_path = path.join(base_path, "{}.log".format(int(time.time())))
        with open(file_path, 'w') as f:
            f.write("{} {}\n".format(4, 4))
            f.write(self.history.__repr__())
            f.flush()
