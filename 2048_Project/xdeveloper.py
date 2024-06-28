
# This file is meant to develop, debug and test parts of the project. Currently testing some improvements on the MCTS

import numpy as np
from time import sleep
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from ai.train import setup_training
from ai.environment import Environment
from game.game_console import print_board
from utils.utilities import gen_key
import json
import os

# Initialize environment and model
env = Environment()
model, json_path, training_info = setup_training()
scores = {"high": 0, "best": 0}

global step_scores, step_bestblock, mcts_scores_up, mcts_scores_left, mcts_scores_down, mcts_scores_right, simulated_steps_up, simulated_steps_left, simulated_steps_down, simulated_steps_right

# Lists to store game data for plotting
# Each step
step_scores = []
step_bestblock = []
mcts_scores_up = []
mcts_scores_left = []
mcts_scores_down = []
mcts_scores_right = []
simulated_steps_up = []
simulated_steps_left = []
simulated_steps_down = []
simulated_steps_right = []


# Calculate the bonus score for the winning strategy
def max_tile_in_corner_value(board):
    """Check if the biggest unique tile is in one of the corners and return its value."""
    board_size = board.shape[0]
    unique_values = np.unique(board)  # Find unique values in the board

    # Check if there are multiple tiles with the max value
    if len(unique_values) != np.count_nonzero(board):
        return 0

    # Find the max value and its positions
    max_value = np.max(board)
    max_indices = np.argwhere(board == max_value)

    # Check if the max value is in one of the corners
    corner_positions = [
        (0, 0),
        (0, board_size - 1),
        (board_size - 1, 0),
        (board_size - 1, board_size - 1),
    ]
    if any(tuple(idx) in corner_positions for idx in max_indices):
        max_position = tuple(max_indices[0])  # Take the first occurrence if multiple
    else:
        return 0

    # Initialize reward calculation
    reward = max_value
    current_value = max_value
    current_position = max_position

    # Sort all unique values in descending order (except the max_value already found)
    unique_values = sorted(set(unique_values) - {max_value}, reverse=True)

    # Iterate through unique values and check neighbors
    for value in unique_values:
        next_indices = np.argwhere(board == value)
        if len(next_indices) == 0:
            break

        # Find if any of the next_indices is a neighbor of current_position
        found_neighbor = False
        for next_position in next_indices:
            if (
                abs(next_position[0] - current_position[0]) <= 1
                and abs(next_position[1] - current_position[1]) <= 1
            ):
                found_neighbor = True
                reward += value
                current_value = value
                current_position = tuple(next_position)
                break

        if not found_neighbor:
            break

    return round(reward // 2)


def slide_and_merge_row_left(row):
    """Slide non-zero tiles to the left and merge equal tiles"""
    new_row = [i for i in row if i != 0]
    changed = len(new_row) != len(row)
    score = 0

    for i in range(len(new_row) - 1):
        if new_row[i] == new_row[i + 1]:
            new_row[i] *= 2
            score += new_row[i]
            new_row[i + 1] = 0
            changed = True

    new_row = [i for i in new_row if i != 0]
    return new_row + [0] * (len(row) - len(new_row)), changed, score


def move_left(board):
    new_board = np.zeros_like(board)
    changed = False
    total_score = 0
    for i in range(4):
        new_row, row_changed, row_score = slide_and_merge_row_left(board[i])
        new_board[i] = new_row
        if row_changed:
            changed = True
            total_score += row_score
    return new_board, changed, total_score


def move_right(board):
    reversed_board = np.fliplr(board)
    new_board = np.zeros_like(board)
    changed = False
    total_score = 0
    for i in range(4):
        new_board[i], row_changed, row_score = slide_and_merge_row_left(
            reversed_board[i]
        )
        if row_changed:
            changed = True
            total_score += row_score
    new_board = np.fliplr(new_board)
    return new_board, changed, total_score


def move_up(board):
    transposed_board = np.transpose(board)
    new_board = np.zeros_like(transposed_board)
    changed = False
    total_score = 0
    for i in range(4):
        new_board[i], row_changed, row_score = slide_and_merge_row_left(
            transposed_board[i]
        )
        if row_changed:
            changed = True
            total_score += row_score
    new_board = np.transpose(new_board)
    return new_board, changed, total_score


def move_down(board):
    transposed_board = np.transpose(board)
    reversed_board = np.fliplr(transposed_board)
    new_board = np.zeros_like(reversed_board)
    changed = False
    total_score = 0
    for i in range(4):
        new_board[i], row_changed, row_score = slide_and_merge_row_left(
            reversed_board[i]
        )
        if row_changed:
            changed = True
            total_score += row_score
    new_board = np.fliplr(new_board)
    new_board = np.transpose(new_board)
    return new_board, changed, total_score


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
    moves = [move_up, move_left, move_down, move_right]
    indices = list(range(len(moves)))
    np.random.shuffle(indices)
    for idx in indices:
        move = moves[idx]
        new_board, valid, new_score = move(board)
        if valid:
            return new_board, valid, new_score, idx
    return board, False, 0, -1


# Improved algo_choose_action function using Monte-Carlo Tree Search
def algo_choose_action(state, current_score, searches_per_move=600, search_length=100):
    first_moves = [move_up, move_left, move_down, move_right]
    scores = np.zeros(4)
    old_state = state.copy()
    direction_counter = np.zeros(4)

    for first_index in range(4):
        first_move = first_moves[first_index]
        first_board, first_valid, first_score = first_move(state)
        if first_valid:
            first_board = add_new_tile(first_board)
            first_total_score = first_score + current_score

            for _ in range(searches_per_move):
                move_number = 1
                search_board = np.copy(first_board)
                is_valid = True
                total_score = first_total_score

                while is_valid and move_number < search_length:
                    search_board, is_valid, move_score, direction = random_move(search_board)
                    direction_counter[direction] += 1
                    if is_valid:
                        search_board = add_new_tile(search_board)
                        total_score += move_score + max_tile_in_corner_value(
                            search_board
                        )
                        move_number += 1

                scores[first_index] += (
                    np.count_nonzero(search_board == 0) * 10
                ) + total_score
        else:
            scores[first_index] = -1

    print(
        f"\nScores: Up {scores[0]} | Left {scores[1]} | Down {scores[2]} | Right {scores[3]}"
    )
    simulated_steps_up.append(direction_counter[0])
    simulated_steps_left.append(direction_counter[1])
    simulated_steps_down.append(direction_counter[2])
    simulated_steps_right.append(direction_counter[3])
    mcts_scores_up.append(scores[0])
    mcts_scores_left.append(scores[1])
    mcts_scores_down.append(scores[2])
    mcts_scores_right.append(scores[3])
    
    # Choose the best move based on scores
    best_move_index = np.argmax(scores)

    # Predict future steps by evaluating the second best move
    second_best_index = np.argsort(scores)[-2]
    second_best_move = first_moves[second_best_index]

    while True:
        best_move = first_moves[best_move_index]
        # Test the best move to check if the board changes
        new_board, is_valid_move, _ = best_move(state)
        boards_equal = np.array_equal(old_state, new_board)
        if not is_valid_move or boards_equal:
            scores[best_move_index] = -1
            best_move_index = np.argmax(scores)
        else:
            break

    return best_move_index


# Function to save dataset (unchanged)
def save_dataset(dataset):
    with open(f"datasets/games_{gen_key()}_monte.json", "w") as f:
        json.dump(dataset, f)

def reset_graphs():
    global step_scores, step_bestblock, mcts_scores_up, mcts_scores_left, mcts_scores_down, mcts_scores_right, simulated_steps_up, simulated_steps_left, simulated_steps_down, simulated_steps_right
    step_scores = []
    step_bestblock = []
    mcts_scores_up = []
    mcts_scores_left = []
    mcts_scores_down = []
    mcts_scores_right = []
    simulated_steps_up = []
    simulated_steps_left = []
    simulated_steps_down = []
    simulated_steps_right = []

# Function to plot game statistics
def plot_game_statistics(game):
    plt.figure(figsize=(14, 10))

    # Plot step scores
    plt.subplot(2, 2, 1)
    plt.plot(step_scores, label='Step Scores', marker='o', linestyle='-')
    plt.xlabel('Steps')
    plt.ylabel('Score')
    plt.title('Score Progression')
    plt.legend()

    # Plot best block values
    plt.subplot(2, 2, 2)
    plt.plot(step_bestblock, label='Best Block', marker='o', linestyle='-')
    plt.xlabel('Steps')
    plt.ylabel('Best Block Value')
    plt.title('Best Block Progression')
    plt.legend()

    # Plot MCTS scores for each direction
    plt.subplot(2, 2, 3)
    plt.plot(mcts_scores_up, label='Up', marker='o', linestyle='-')
    plt.plot(mcts_scores_left, label='Left', marker='o', linestyle='-')
    plt.plot(mcts_scores_down, label='Down', marker='o', linestyle='-')
    plt.plot(mcts_scores_right, label='Right', marker='o', linestyle='-')
    plt.xlabel('Steps')
    plt.ylabel('MCTS Scores')
    plt.title('MCTS Scores Progression')
    plt.legend()

    # Plot simulated steps for each direction
    plt.subplot(2, 2, 4)
    plt.plot(simulated_steps_up, label='Up', marker='o', linestyle='-')
    plt.plot(simulated_steps_left, label='Left', marker='o', linestyle='-')
    plt.plot(simulated_steps_down, label='Down', marker='o', linestyle='-')
    plt.plot(simulated_steps_right, label='Right', marker='o', linestyle='-')
    plt.xlabel('Steps')
    plt.ylabel('Simulated Steps')
    plt.title('Simulated Steps Progression')
    plt.legend()

    plt.tight_layout()
    save_path = "dev_plots"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = os.path.join(save_path, f"Game_{game}_plot.png")
    plt.savefig(file_path, bbox_inches="tight")
    print(f"Plot wurde gespeichert unter: {file_path}")
    plt.show()



# Function to play the game with the AI
def ai_play_game(num_games=1):
    save_data = []
    highscore = 0
    best_block = 0
    games = 0
    try:
        for _ in range(num_games):
            state = env.reset()
            env.game.score = 0
            done = False
            steps = 0
            while not done:
                steps += 1
                if env.game.highscore > scores["high"]:
                    scores["high"] = env.game.highscore

                if env.game.bestblock > scores["best"]:
                    scores["best"] = env.game.bestblock

                if env.game.score > highscore:
                    highscore = env.game.score
                if env.game.bestblock > best_block:
                    best_block = env.game.bestblock

                action = algo_choose_action(state, env.game.score)
                state, _, done = env.step(action)
                print(f">> Highscore: [ {highscore} ]  || BestBlock: [ {best_block} ]")
                print(f">> Is DONE: {done}")
                env.render(scores, training_info["traintime"])
                choice = ["UP", "LEFT", "DOWN", "RIGHT"][action]
                print(f">> {choice}")
                save_data.append([state.tolist(), choice])
                
                step_scores.append(env.game.score)
                step_bestblock.append(env.game.bestblock)
                
                # sleep(1)
            
            plot_game_statistics(games)  # Plot statistics after each game
            games += 1
            
            print("Game over. Final score:", env.game.score)
            save_dataset(save_data)
    except KeyboardInterrupt:
        save_dataset(save_data)
        return


def save_dataset(dataset):
    with open(f"datasets/games_{gen_key()}_monte.json", "w") as f:
        json.dump(dataset, f)


# Play test games
ai_play_game(1)
