import numpy as np
from time import sleep
from ai.train import setup_training
from ai.environment import Environment
from game.game_console import print_board
from utils.utilities import gen_key
import json

# Initialize environment and model
env = Environment()
model, json_path, training_info = setup_training()
scores = {"high": 0, "best": 0}

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
def algo_choose_action(state, searches_per_move=300, search_length=50):
    first_moves = [move_up, move_left, move_down, move_right]
    scores = np.zeros(4)
    old_state = state.copy()

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
            scores[first_index] = -1  # Penalize invalid moves
    print(f"\n- - - - - -- - - - - - - - - - - - -\nScores: Up {scores[0]} | Left {scores[1]} | Down {scores[2]} | Right {scores[3]}")
    # Choose the best move based on scores
    best_move_index = np.argmax(scores)
    
    while True:
        best_move = first_moves[best_move_index]
        # Test the best move to check if the board changes
        new_board, is_valid_move = best_move(state)
        boards_equal = np.array_equal(old_state, new_board)
        if not is_valid_move or boards_equal:
            scores[best_move_index] = -1
            best_move_index = np.argmax(scores)
        else:
            break

    return best_move_index

# Function to play the game with the AI
def ai_play_game(num_games=1):
    save_data = []
    highscore = 0
    best_block = 0
    try:
        for _ in range(num_games):
            state = env.reset()
            env.game.score = 0
            done = False
            steps = 0
            while not done:
                steps += 1
                #if steps > 500:
                #    print("Game finished at 500 moves")
                #    break

                if env.game.highscore > scores["high"]:
                    scores["high"] = env.game.highscore

                if env.game.bestblock > scores["best"]:
                    scores["best"] = env.game.bestblock

                if env.game.score > highscore:
                    highscore = env.game.score
                if env.game.bestblock > best_block:
                    best_block = env.game.bestblock

                action = algo_choose_action(state)
                state, _, done = env.step(action)
                print(f">> Highscore: [ {highscore} ]  || BestBlock: [ {best_block} ]")
                print(f">> Is DONE: {done}")
                env.render(scores, training_info["traintime"])
                choice = ["UP", "LEFT", "DOWN", "RIGHT"][action]
                print(f">> {choice}")
                save_data.append([state.tolist(), choice])
                #sleep(1)
            print("Game over. Final score:", env.game.score)
            save_dataset(save_data)
    except KeyboardInterrupt:
        save_dataset(save_data)
        return
    except Exception as e:
        print(f"Error: {e}")
        input("\n\nConfirm error")
        return

def save_dataset(dataset):
    with open(f"datasets/games_{gen_key()}_monte.json", "w") as f:
        json.dump(dataset, f)

# Play a single game to test
ai_play_game(20)
