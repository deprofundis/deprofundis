import numpy as np

import Tkinter


class Generator(object):
    def __init__(self, rows=20, cols=20):
        self.cols = cols
        self.rows = rows
        self.board = None

    def seed(self, offset=5):
        """
        Initializes a board
        @param row_num: Number of rows
        @param col_num: Number of cilumns
        @param offset: Offset for the index where to start computiong the initial values
        @return: Initialized board
        """
        self.board = np.zeros(shape=(self.rows, self.cols),dtype=np.int8)
        for row in range(offset, self.rows - offset):
            for col in range(offset, self.cols - offset):
                self.board[row, col] = (np.random.uniform() > 0.5)

        return self.board

    def generate_series(self, steps=5,burnin=5,plot=False):
        """
        Generate a sequence of boards.
        @param steps: Number of steps to simulate
        @return: Two-dimensional array with each row being one step
        """
        assert (self.board is not None)

        # array which holds each single game
        board_collection = np.zeros(shape=(steps, self.rows * self.cols))
        for steps in range(steps+burnin):
            self.board = self.generate_step()
            # store into return vector, after burn-in phase is over
            if steps >= burnin:
                board_collection[steps] = np.reshape(self.board, newshape=self.rows * self.cols)

        return board_collection


    def generate_step(self):
        """
        Helper function to create a next step board. Rules:

        1. Any live cell with fewer than two live neighbours dies, as if caused by under-population.
        2. Any live cell with two or three live neighbours lives on to the next generation.
        3. Any live cell with more than three live neighbours dies, as if by overcrowding.
        4. Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.

        @return: Board of the next step
        """
        assert (self.board is not None)

        next_board = np.zeros(shape=self.board.shape)
        for row in range(self.rows):
            for col in range(self.cols):
                cell = self.board[row, col]
                alive_neighbors = self.num_alive_neighbors(row, col)

                # Alter cell states
                if cell == 1:
                    next_board[row, col] = self.alter_alive_cell(alive_neighbors)
                elif cell == 0:
                    next_board[row, col] = self.alter_dead_cell(alive_neighbors)
                else:
                    raise ValueError('The value of a cell can only be 1 or 0.')

        return next_board

    def alter_alive_cell(self, alive_neighbors):
        """
        Applies rules for an alive cell.
        @param alive_neighbors: number of alive neighbors
        @return: Alive (1) or dead(0)
        """
        if alive_neighbors < 2 or alive_neighbors > 3:
            return 0
        else:
            return 1

    def alter_dead_cell(self, alive_neighbors):
        """
        Applies rules to a dead cell.
        @param alive_neighbors: number of alive neighbors
        @return: Alive (1) or dead (0)
        """
        if alive_neighbors is 3:
            return 1
        else:
            return 0

    def num_alive_neighbors(self, row, col):
        """
        Computes the alive neighbors using a torus approach (if an alive element is at the edge of the game we switch
        to the opposite side.
        @param row: Row location of the element
        @param col: Column location of the element
        @return: number of alive elements
        """
        alive = 0
        row_max = 0 if (row + 1) > self.rows - 1 else row + 1
        col_max = 0 if (col + 1) > self.cols - 1 else col + 1
        for row_idx in (row - 1, row_max):
            for col_idx in (col - 1, col_max):
                alive += (self.board[row_idx, col_idx] > 0)

        return alive