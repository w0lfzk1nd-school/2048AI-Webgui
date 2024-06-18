import random
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.utilities import print_and_log, log, get_current_time


def print_multi_board(old_board, new_board):
    max_rows = max(len(old_board), len(new_board))
    txt = ">> Old Board  |  New Board\n"
    for i in range(max_rows):
        if i < len(old_board):
            txt += "+----" * len(old_board[i]) + "+ |>> "
        else:
            txt += " " * (5 * len(old_board[0]) + 7)

        if i < len(new_board):
            txt += "+----" * len(new_board[i]) + "+\n"
        else:
            txt += "\n"

        if i < len(old_board):
            txt += (
                "|"
                + "|".join(f"{num:4d}" if num > 0 else "    " for num in old_board[i])
                + "| |>> "
            )
        else:
            txt += " " * (5 * len(old_board[0]) + 7)

        if i < len(new_board):
            txt += (
                "|"
                + "|".join(f"{num:4d}" if num > 0 else "    " for num in new_board[i])
                + "|\n"
            )
        else:
            txt += "\n"

    txt += "+----" * len(old_board[0]) + "+ |>> "
    txt += "+----" * len(new_board[0]) + "+"

    print_and_log(txt)


def print_board(board):
    txt = ""
    for row in board:
        txt += "+----" * len(row) + "+\n"
        txt += "|".join(f"{num:4d}" if int(num) > 0 else "    " for num in row) + "|\n"
    txt += "+----" * len(row) + "+"
    print_and_log(txt)


class Game2048:
    def __init__(self, size=4):
        self.size = size
        self.board = [[0] * size for _ in range(size)]
        self.score = 0
        self.highscore = 0
        self.bestblock = 0
        self.add_new_tile()
        self.add_new_tile()

    def reset(self):
        size = 4
        self.board = [[0] * size for _ in range(size)]
        self.score = 0
        self.highscore = 0
        self.bestblock = 0
        self.add_new_tile()
        self.add_new_tile()

    def get_board(self):
        return self.board

    def add_new_tile(self):
        empty_cells = [
            (x, y)
            for x in range(self.size)
            for y in range(self.size)
            if self.board[x][y] == 0
        ]
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x][y] = random.choice([2, 4])

    def move(self, direction):
        old_board = [row[:] for row in self.board]

        def merge(row):
            non_zero = [i for i in row if i != 0]
            merged = []
            i = 0
            while i < len(non_zero):
                if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                    merged_score = non_zero[i] * 2
                    self.score += merged_score

                    if self.score > self.highscore:
                        self.highscore = self.score

                    if merged_score > self.bestblock:
                        self.bestblock = merged_score

                    merged.append(non_zero[i] * 2)
                    i += 2
                else:
                    merged.append(non_zero[i])
                    i += 1
            return merged + [0] * (self.size - len(merged))

        def transpose(board):
            return [list(row) for row in zip(*board)]

        def reverse(board):
            return [row[::-1] for row in board]

        if direction == "w" or direction == 0:
            self.board = transpose(self.board)
            self.board = [merge(row) for row in self.board]
            self.board = transpose(self.board)
        elif direction == "s" or direction == 2:
            self.board = transpose(self.board)
            self.board = reverse(self.board)
            self.board = [merge(row) for row in self.board]
            self.board = reverse(self.board)
            self.board = transpose(self.board)
        elif direction == "a" or direction == 1:
            self.board = [merge(row) for row in self.board]
        elif direction == "d" or direction == 3:
            self.board = reverse(self.board)
            self.board = [merge(row) for row in self.board]
            self.board = reverse(self.board)

        readable_action = ["[UP]", "[LEFT]", "[DOWN]", ["RIGHT"]][direction]

        if self.board != old_board:
            print_and_log(
                f"- - - - - - - - - - - - - - - - - - -\n{get_current_time()} Move\n>> Chose Ation:     {readable_action}"
            )
            self.add_new_tile()
            print_multi_board(old_board, self.board)
            
        return self.is_game_over()

    def is_game_over(self):
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x][y] == 0:
                    return False
                if x < self.size - 1 and self.board[x][y] == self.board[x + 1][y]:
                    return False
                if y < self.size - 1 and self.board[x][y] == self.board[x][y + 1]:
                    return False
        return True
