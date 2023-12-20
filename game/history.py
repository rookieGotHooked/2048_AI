import numpy as np

class History:
    """
    This class represents an History of a 2048 game with:
        - The different scores over time
        - The different Grid states over time
        - The different directions that have been played over time
    """

    def __init__(self, nb_rows, nb_columns):
        """
        Init method to initialize a new History object

        @param nb_rows: the number of rows of the Grid for that Game
        @type nb_rows: int
        @param nb_columns: the number of columns of the Grid for that Game
        @type nb_columns: int
        """
        self.nb_rows = nb_rows
        self.nb_columns = nb_columns
        self.grid_history = list()
        self.score_history = list()
        self.direction_state_history = list()
        self.direction_index_history = list()

    def __repr__(self):
        """
        An utility method to get the string representation of this History object

        @return: the entire history of a Game
        @rtype: str
        """
        str_to_return = ""
        for i in range(len(self.grid_history)):
            str_to_return += "{} ".format(i)
            str_to_return += "{} ".format(self.score_history[i])
            str_to_return += str(self.grid_history[i])
            print(self.direction_state_history)
            str_to_return += " {}".format(self.direction_state_history[i].value)
            str_to_return += " [{}]".format(self.direction_index_history[i])
            str_to_return += "\n"
        return str_to_return

    def add_grid_state(self, t_str_state, score):
        """
        Method to add a new Grid snapshot to this History object
        A new Grid state can mean a change in score too so we keep track of it too

        @param t_str_state: the new Grid state to add
        @type t_str_state: str
        @param score: the score associated with that Grid state
        @type score: int
        """
        self.grid_history.append(t_str_state)
        self.score_history.append(score)

    def add_direction_or_state(self, direction_or_state, index_choice):
        """
        Method to add a new direction/state (if win or loss) to this History

        @param direction_or_state: the direction that has been played or the final state reached (win/loss)
        @type direction_or_state: Constants.Directions or Constants.States
        @param index_choice: the index of the chosen direction (0 was the first choice, 3 was the last choice)
        @type index_choice: int
        """
        self.direction_state_history.append(direction_or_state)
        self.direction_index_history.append(index_choice)

    def something_moved(self, previous_state):
        """
        Method to determine if at least one tile has moved between two Grid snapshots (current, previous)

        @param previous_state: the previous Grid state (or snapshot) in inline str representation format
        @type previous_state: str
        @return: whether at least one tile has moved compared to the previous Grid snapshot
        @rtype: bool
        """
        state_a = self.from_string(previous_state, self.nb_rows, self.nb_columns)
        state_b = self.from_string(self.grid_history[-1], self.nb_rows, self.nb_columns)
        return not np.array_equal(state_a, state_b)

    @staticmethod
    def from_string(t_str_grid, nb_rows, nb_columns):
        """
        Utility method to convert an inline string representation of a Grid into a numpy array (or matrix)

        @param t_str_grid: an inline string representation of a Grid
        @type t_str_grid: str
        @param nb_rows: the number of rows of the final numpy array
        @type nb_rows: int
        @param nb_columns: the number of columns of the final numpy array
        @type nb_columns: int
        @return: a numpy array object corresponding to the inline grid representation
        @rtype: ndarray
        """
        t_list = [int(i) for i in t_str_grid.strip().split(' ')]
        return np.reshape(t_list, (nb_rows, nb_columns))