import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ai.environment import Environment
from utils.utilities import print_and_log, log, get_current_time

# Initialize environment and model
env = Environment()


def print_board(board):
    txt = ""
    for row in board:
        txt += "+----" * len(row) + "+\n"
        txt += "|".join(f"{num:4d}" if int(num) > 0 else "    " for num in row) + "|\n"
    txt += "+----" * len(row) + "+"
    print_and_log(txt)


def slide_and_merge_row_left(row):
    """Slide non-zero tiles to the left and merge equal tiles"""
    new_row = [i for i in row if i != 0]
    changed = len(new_row) != len(row)  # Track if the row changes in length

    for i in range(len(new_row) - 1):
        if new_row[i] == new_row[i + 1]:
            new_row[i] *= 2
            new_row[i + 1] = 0
            changed = True  # Mark as changed when a merge occurs

    new_row = [i for i in new_row if i != 0]
    return new_row + [0] * (len(row) - len(new_row)), changed


def move_left(board):
    new_board = np.zeros_like(board)
    changed = False
    for i in range(4):
        new_board[i], row_changed = slide_and_merge_row_left(board[i])
        if row_changed:
            changed = True
    return new_board, changed


def move_right(board):
    reversed_board = np.fliplr(board)
    new_board = np.zeros_like(board)
    changed = False
    for i in range(4):
        new_board[i], row_changed = slide_and_merge_row_left(reversed_board[i])
        if row_changed:
            changed = True
    new_board = np.fliplr(new_board)
    return new_board, changed


def move_up(board):
    transposed_board = np.transpose(board)
    new_board = np.zeros_like(transposed_board)
    changed = False
    for i in range(4):
        new_board[i], row_changed = slide_and_merge_row_left(transposed_board[i])
        if row_changed:
            changed = True
    new_board = np.transpose(new_board)
    return new_board, changed


def move_down(board):
    transposed_board = np.transpose(board)
    reversed_board = np.fliplr(transposed_board)
    new_board = np.zeros_like(reversed_board)
    changed = False
    for i in range(4):
        new_board[i], row_changed = slide_and_merge_row_left(reversed_board[i])
        if row_changed:
            changed = True
    new_board = np.fliplr(new_board)
    new_board = np.transpose(new_board)
    return new_board, changed


# Function to add a new tile to the board
def add_new_tile(board):
    empty_positions = [(i, j) for i in range(4) for j in range(4) if board[i][j] == 0]
    if empty_positions:
        i, j = empty_positions[np.random.choice(len(empty_positions))]
        board[i][j] = 2 if np.random.rand() < 0.9 else 4
    return board


# Function to check if any move is possible
def any_moves_possible(board):
    for move in [move_left, move_right, move_up, move_down]:
        _, valid = move(board)
        if valid:
            return True
    return False


# Function to make a random move
def random_move(board):
    moves = [move_up, move_down, move_left, move_right]
    np.random.shuffle(moves)
    for move in moves:
        new_board, valid = move(board)
        if valid:
            return new_board, valid
    return board, False


# Improved algo_choose_action function using Monte-Carlo Tree Search
def monte_predict(state, searches_per_move=500, search_length=10):
    first_moves = [move_up, move_left, move_down, move_right]
    scores = np.zeros(4)
    old_state = state.copy()
    print_and_log(
        f"\n- - - - - - - - - - - - - - - - - - - - - -\n>> ({get_current_time()}) START PREDICTING NEXT MOVE"
    )

    for first_index in range(4):
        first_move = first_moves[first_index]
        first_board, first_valid = first_move(state)
        if first_valid:
            first_board = add_new_tile(first_board)

            for _ in range(searches_per_move):
                move_number = 1
                search_board = np.copy(first_board)
                is_valid = True

                while is_valid and move_number < search_length:
                    search_board, is_valid = random_move(search_board)
                    if is_valid:
                        search_board = add_new_tile(search_board)
                        move_number += 1
                    scores[first_index] += np.count_nonzero(search_board == 0)
        else:
            scores[first_index] = -1

    best_move_index = np.argmax(scores)

    while True:
        best_move = first_moves[best_move_index]
        new_board, is_valid_move = best_move(state)
        boards_equal = np.array_equal(old_state, new_board)
        if not is_valid_move or boards_equal:
            scores[best_move_index] = -1
            best_move_index = np.argmax(scores)
        elif np.sum(scores) <= 0 or not any_moves_possible(new_board):
            break
        else:
            break
    readable_action = ["[UP]", "[LEFT]", "[DOWN]", ["RIGHT"]][best_move_index]
    print_and_log(
        f"\n-- >> Predicted Move: {readable_action}\nScores: Up {scores[0]} | Left {scores[1]} | Down {scores[2]} | Right {scores[3]}"
    )
    print_board(new_board)
    print_and_log("\n- - - - - - - - - - - - - - - - - - -")

    return {"predicted": best_move_index, "scores": scores}
