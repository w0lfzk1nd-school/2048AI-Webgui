import numpy as np
import json
import random
import os
import questionary
import datetime
from time import sleep
from game.game_console import print_board
from game.game_engine import Game2048
from ai.ollama import OllamaSolve
from ai.monte_carlo import monte_predict
from ai.environment import Environment
import multiprocessing as mp
from functools import partial
from ai.model import create_model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from utils.utilities import (
    select_model,
    select_dataset,
    gen_key,
    get_clean_key,
    print_and_log,
    form_time,
    log,
)


def default_converter(o):
    if isinstance(o, np.integer):
        return int(o)
    elif isinstance(o, np.floating):
        return float(o)
    elif isinstance(o, np.ndarray):
        return o.tolist()
    else:
        return str(o)


def has_valid_move(board):
    for row in range(4):
        for col in range(4):
            if board[row][col] == 0:
                return True

    for row in range(4):
        for col in range(4):
            current_value = board[row][col]
            if col < 3 and board[row][col + 1] == current_value:
                return True
            if row < 3 and board[row + 1][col] == current_value:
                return True
    return False


def is_single_number(output):
    try:
        float_output = float(output)
        if float_output >= 4 or float_output < 0:
            return False
        return True
    except ValueError:
        return False


def game_console(num_games=1):
    game_key, game_data, training_info = setup_dataset()

    for _ in range(num_games):
        print(f">> Game Key: {game_key}")
        choice = questionary.select(
            "Choose an option:",
            choices=["Play the game yourself",
                     "Generate random game boards", "Exit"],
        ).ask()

        if choice == "Play the game yourself":
            game_data = play_game(game_data, training_info)
            filename = questionary.text(
                "Enter dataset filename to save:",
                default=f"{game_key}_hooman_game.json",
            ).ask()
            save_dataset(game_data, filename)
            print(f">> Dataset saved. {filename}")

        elif choice == "Generate random game boards":
            num_boards = questionary.text(
                "Number of random game boards:", default="10"
            ).ask()
            game_data = generate_random_boards(int(num_boards))
            filename = questionary.text(
                "Enter dataset filename to save:",
                default=f"{game_key}_random_boards.json",
            ).ask()
            save_dataset(game_data, filename)
            print(f">> Dataset saved. {filename}")

    save_dataset(game_data, filename)
    print(f">> Final Datasave complete")
    return game_data


def play_game(game_data, training_info):
    global highscore
    env = Environment()
    game = env.game

    highscore = 0
    bestscore = 0
    steps = 0
    state = env.reset()

    train_start = datetime.datetime.now()

    while not game.is_game_over():

        if env.game.highscore > highscore:
            highscore = env.game.highscore

        if env.game.bestblock > bestscore:
            bestscore = env.game.bestblock

        env.render({"high": highscore, "best": bestscore},
                   training_info["traintime"])

        move = input("Enter move (w, a, s, d): ").strip().lower()
        if not move:
            print("Invalid move. Please only use w, a, s, or d.")
            continue
        elif move == "exit":
            break
        elif move in ["w", "a", "s", "d"]:
            action = ["w", "a", "s", "d"].index(move)

            # Überprüfen, ob die Einträge in game.board NumPy-Arrays sind und konvertieren
            if isinstance(game.board, np.ndarray):
                board_list = game.board.tolist()
            else:
                board_list = [
                    row.tolist() if isinstance(row, np.ndarray) else row
                    for row in game.board
                ]

            game_data.append((board_list, action))

            next_state, reward, done = env.step(action)
            game.board = next_state
            steps += 1
            current_time = datetime.datetime.now()
            time_trained = (current_time - train_start).total_seconds() * 1000
            training_info["traintime"] += round(time_trained, 3)
        elif move == "restart":
            # os.system("cls")
            game = Game2048()
            game_data = []
        else:
            print("Invalid move. Please only use w, a, s, or d.")

    print("Game over! Final board:")
    print_board(game.board)

    current_time = datetime.datetime.now()
    time_trained = (current_time - train_start).total_seconds() * 1000
    training_info["traintime"] += round(time_trained, 3)
    env.runtime_history.append(round(time_trained, 3))
    print_and_log(
        f"Zeit für diese Episode: {round(time_trained, 3)} ms ({form_time(round(time_trained, 3))}), Gesamttrainingszeit: {training_info['traintime']} ms (({form_time(training_info['traintime'])}))"
    )
    update_training_info(training_info, steps, env.game.score)

    return game_data


def setup_dataset():
    load_existing = (
        input("Möchten Sie ein existierendes Dataset laden? (y/n): ").lower() == "y"
    )

    if load_existing:
        path_to_dataset = select_dataset()
        model_key, game_data, training_info = load_dataset(path_to_dataset)
        json_path = f"datasets/{model_key}-data.json"
        with open(json_path, "r") as f:
            training_info = json.load(f)
    else:
        model_key = gen_key()
        game_data = []
        json_path = f"datasets/{model_key}-data.json"
        training_info = []  # init_dataset_info(json_path)

    return model_key, game_data, training_info


def init_dataset_info(json_path):
    training_info = {
        "model_name": get_clean_key(json_path.split("/")[-1]),
        "model_start": datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S"),
        "last_training": "",
        "traintime": 0,
        "total_train_steps": 0,
        "total_games": 0,
        "avg_points": 0,
        "highscore": 0,
    }
    with open(json_path, "w") as f:
        json.dump(training_info, f)
    return training_info


def update_training_info(training_info, steps, score):
    training_info["total_train_steps"] += steps
    training_info["total_games"] += 1
    training_info["avg_points"] = (
        training_info["avg_points"] *
        (training_info["total_games"] - 1) + score
    ) / training_info["total_games"]
    training_info["highscore"] = highscore


def generate_random_boards(num_boards):
    game_data = []

    for _ in range(num_boards):
        state = generate_valid_board()
        print_board(state)
        while True:
            action = questionary.text("Enter your action (W, A, S, D):").ask()
            if action == "exit" or not action:
                return game_data
            elif action.lower() in ["w", "a", "s", "d"]:
                action = ["w", "a", "s", "d"].index(
                    action)  # Convert action to number
                if isinstance(state, np.ndarray):
                    board_list = state.tolist()
                else:
                    board_list = [
                        row.tolist() if isinstance(row, np.ndarray) else row
                        for row in state
                    ]

                game_data.append((board_list, action))

                break
            else:
                print("Invalid input. Please choose a valid action.")

    return game_data


def create_games_dataset(num_games=1000, max_moves=500, filename="gen_games.json"):
    dataset = []
    moves = ["Up", "Left", "Down", "Right"]
    env = Environment()
    use_ollama = questionary.confirm(
        "Do you want to process the imput from ollama LLM?"
    ).ask()
    if use_ollama:
        ollama_env = OllamaSolve()

    highscore = 0
    best_block = 0
    games = 0

    for _ in range(num_games):
        state = env.reset()
        if use_ollama:
            ollama_env.reset_chat()
        game_data = []
        done = False
        move_count = 0
        score = 0
        highest_block = 0

        while not done and move_count < max_moves:
            info_txt = f"""<- - - - - - - - - - - - ->
>> Games: {games}/{num_games}
>> Move: {move_count}
>> Highscore: [{highscore}] | Best Block: [{best_block}]
>> Current Score: [{score}] | Block: {highest_block}
<- - - - - - - - - - - - ->
"""

            if use_ollama:
                print(f">> Ollama solving Board")
                for row in state:
                    print("+----" * len(row) + "+")
                    log("+----" * len(row) + "+")
                    print(
                        "|".join(f"{num:4d}" if num >
                                 0 else "    " for num in row)
                        + "|"
                    )
                    log(
                        "|".join(f"{num:4d}" if num >
                                 0 else "    " for num in row)
                        + "|"
                    )
                print("+----" * len(row) + "+")
                log("+----" * len(row) + "+")
                while True:
                    try:
                        action = ollama_env.solve_board(f"{state}")
                    except Exception as e:
                        print(f">> OLLAMA ERROR: {e}")
                        action = random.randrange(4)
                    if is_single_number(action):
                        break
                    print(f"Output no INT, ollama env reset\n\n{action}")
                    ollama_env.reset_chat()
                action = int(action)
            else:
                action = random.randrange(4)
            print_and_log(f"Action: {action} : {moves[action]}")
            log(f"Action: {action} : {moves[action]}")
            next_state, reward, done = env.step(action)
            game_data.append((state.tolist(), action))
            state = next_state
            move_count += 1
            score += reward
            highest_block = np.max(state)

            if use_ollama and move_count % 5 == 0:
                print_and_log("Resetting Ollama chat...")
                log("Resetting Ollama chat due to 5x Limit...")
                ollama_env.reset_chat()
        print(">> IN DATASET: Game finished")
        games += 1
        if score > highscore:
            highscore = score
        if highest_block > best_block:
            best_block = highest_block

        log(
            f"<- - - - - - - - - - - - ->\>> Game {games}/{num_games} Final Report:\M>> Moves: {move_count}\nCurrent Score: [{score}] | Block: {highest_block}\n<- - - - - - - - - - - - ->"
        )
        if use_ollama:
            print(">> Sleeping")
            sleep(30)
            print(">> AAAAH I WOKE UP")
            dataset.extend(game_data)

    save_dataset(dataset, filename)
    print(
        f"Games dataset generated with {len(dataset)} samples and saved to {filename}"
    )


def create_random_dataset(num_samples=10000, filename="gen_random.json"):
    dataset = []

    for _ in range(num_samples):
        state = generate_valid_board()
        action = random.randrange(4)
        dataset.append((state, action))
        # dataset.append((state))

    save_dataset(dataset, filename)
    print(
        f"Random dataset generated with {len(dataset)} samples and saved to {filename}"
    )


def generate_valid_board():
    def generate_early_game_board():
        board = np.zeros((4, 4), dtype=int)
        for i in range(4):
            for j in range(4):
                if random.random() < 0.7:
                    board[i][j] = 0
                else:
                    board[i][j] = random.choice(
                        [2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 16]
                    )
        return board

    def generate_mid_game_board():
        board = np.zeros((4, 4), dtype=int)
        for i in range(4):
            for j in range(4):
                if random.random() < 0.3:
                    board[i][j] = 0
                else:
                    board[i][j] = random.choice(
                        [
                            2,
                            2,
                            2,
                            4,
                            4,
                            4,
                            8,
                            8,
                            8,
                            16,
                            16,
                            16,
                            32,
                            32,
                            64,
                            64,
                            128,
                            256,
                            512,
                        ]
                    )
        return board

    def generate_late_game_board():
        board = np.zeros((4, 4), dtype=int)
        for i in range(4):
            for j in range(4):
                board[i][j] = random.choice(
                    [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])

        num_pairs = 2
        while num_pairs > 0:
            i1, j1 = random.randint(0, 3), random.randint(0, 3)
            if board[i1][j1] != 0:
                value = board[i1][j1]
                neighbors = [
                    (i1 + di, j1 + dj)
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    if 0 <= i1 + di < 4 and 0 <= j1 + dj < 4
                ]
                for ni, nj in neighbors:
                    if board[ni][nj] != value:
                        board[ni][nj] = value
                        num_pairs -= 1
                        break
        return board

    game_phase = random.choice(["early", "mid", "late"])
    if game_phase == "early":
        return generate_early_game_board().tolist()
    elif game_phase == "mid":
        return generate_mid_game_board().tolist()
    else:
        return generate_late_game_board().tolist()


def save_dataset(dataset, filename):
    with open(f"datasets/{filename}", "w") as f:
        json.dump(dataset, f, default=default_converter)


def load_dataset(filename):
    cleankey = get_clean_key(filename)
    with open(f"datasets/{filename}", "r") as f:
        dataset = json.load(f)
    if os.path.exists(f"datasets/{cleankey}-data.json"):
        with open(f"datasets/{cleankey}-data.json", "r") as f:
            training_info = json.load(f)
    else:
        training_info = []
    return cleankey, dataset, training_info


def preprocess_dataset(dataset):
    states = np.array([np.array(data[0]) for data in dataset])
    actions = np.array([data[1] for data in dataset])
    return states, actions


def augment_data(board, action):
    augmented_data = []

    # Original
    augmented_data.append((board, action))

    # Flip Left-Right
    flipped_lr_board = np.fliplr(board)
    flipped_lr_action = 3 if action == 1 else 1 if action == 3 else action
    augmented_data.append((flipped_lr_board, flipped_lr_action))

    # Flip Up-Down
    flipped_ud_board = np.flipud(board)
    flipped_ud_action = 2 if action == 0 else 0 if action == 2 else action
    augmented_data.append((flipped_ud_board, flipped_ud_action))

    if isinstance(action, str):
        try:
            action = int(action)
        except:
            action = 0

    # Rotations
    for k in range(1, 4):
        rotated_board = np.rot90(board, k)
        rotated_action = (action + k) % 4
        augmented_data.append((rotated_board, rotated_action))

    return augmented_data


def generate_augmented_dataset(dataset):
    augmented_dataset = []
    for board, action in dataset:
        augmented_dataset.extend(augment_data(board, action))
    return augmented_dataset


def user_finetune_augemntation(augmented_set):
    aug_data_len = len(augmented_set)
    input(f">> Augmented Dataset has len: {aug_data_len}")
    new_augmented = []
    try:
        for turn in augmented_set:
            print(f"new_augm: {new_augmented}")
            board = turn[0]
            action = turn[1]
            form_action = ["w", "a", "s", "d"][action]
            print_board(board)
            print(f"\n>> Action: {action} | {form_action}")
            while True:
                action = questionary.text(
                    "Enter new action (W, A, S, D) or ENTER to keep value or 'exit' to return:"
                ).ask()
                if action == "exit":
                    return new_augmented

                elif action.lower() in ["w", "a", "s", "d"]:
                    new_action = ["w", "a", "s", "d"].index(action)
                    if new_action == action:
                        break
                    else:
                        turn[1] = new_action
                        new_augmented.append(turn)
                    break
                elif action == None:
                    break
                else:
                    print("Invalid input. Please choose a valid action. Or ENTER to ")
    except KeyboardInterrupt:
        return new_augmented
    except Exception as e:
        print(f"Error: {e}")
        return new_augmented


def train_model_with_dataset(model, dataset, epochs=10, batch_size=32):
    if len(dataset) == 3:
        key, dataset_data, training = dataset
    else:
        dataset_data = dataset
    print(f"Dataset len: {len(dataset_data)}")
    states, actions = preprocess_dataset(dataset_data)

    # One-hot encode actions
    actions_one_hot = np.zeros((actions.size, 4))
    actions_one_hot[np.arange(actions.size), actions] = 1

    # Train the model with the original dataset
    print("Training with original dataset...")
    model.fit(states, actions_one_hot, epochs=epochs, batch_size=batch_size)

    if questionary.confirm("Do you want to train the augmented dataset too?").ask():
        # -- Generate and preprocess the augmented dataset
        augmented_dataset = generate_augmented_dataset(dataset_data)
        augmented_states, augmented_actions = preprocess_dataset(
            augmented_dataset)

        # -- One-hot encode augmented actions
        augmented_actions_one_hot = np.zeros((augmented_actions.size, 4))
        augmented_actions_one_hot[
            np.arange(augmented_actions.size), augmented_actions
        ] = 1

        # -- Train the model with the augmented dataset
        print("Training with augmented dataset...")
        model.fit(
            augmented_states,
            augmented_actions_one_hot,
            epochs=epochs,
            batch_size=batch_size,
        )


def create_shuffled_copy():
    chosen_dataset = select_dataset()
    key, dataset, traininginfo = load_dataset(chosen_dataset)

    shuffled_dataset = dataset.copy()
    random.shuffle(shuffled_dataset)
    sort_key = gen_key()

    shuffled_name = f"{key}_{sort_key}_shuffled.json"
    save_dataset(shuffled_dataset, shuffled_name)

    print(f"Shuffled Dataset saved to: {shuffled_name}")


def split_dataset():
    chosen_dataset = select_dataset()
    key, dataset, traininginfo = load_dataset(chosen_dataset)

    print(f"Original Dataset Size: {len(dataset)}")
    num_parts = int(
        input("Enter the number of parts to split the dataset into (1-10): ")
    )

    if num_parts <= 0 or num_parts > 10:
        print("Number of parts should be between 1 and 10.")
        return

    part_size = len(dataset) // num_parts
    remainder = len(dataset) % num_parts

    split_datasets = []
    start_index = 0

    for part in range(num_parts):
        part_length = part_size + 1 if part < remainder else part_size
        split_datasets.append(dataset[start_index: start_index + part_length])
        start_index += part_length

    for i, split_data in enumerate(split_datasets):
        part_name = f"{key}_part{i+1}_{num_parts}_split.json"
        save_dataset(split_data, part_name)
        print(f"Split Dataset part {i+1} saved to: {part_name}")

    print(f"Dataset split into {num_parts} parts.")


def resize_dataset():
    chosen_dataset = select_dataset()
    key, dataset, traininginfo = load_dataset(chosen_dataset)

    print(f"Original Dataset Size: {len(dataset)}")
    new_size = int(
        input(
            f"Enter the new size of the dataset: (between 1 and {len(dataset)}):")
    )
    print(f"{new_size}, {type(new_size)}, {len(dataset)}, {type(len(dataset))}")

    if new_size >= len(dataset):
        print("New size should be smaller than the original size.")
        return

    selection_method = input(
        "Select data selection method (Original Sequence 'o' or Random Order 'r'): "
    ).lower()
    if selection_method not in ["o", "r"]:
        print("Invalid selection method.")
        return

    if selection_method == "o":
        new_dataset = dataset * (new_size // len(dataset))
        remainder = new_size % len(dataset)
        new_dataset += dataset[:remainder]
    else:
        new_dataset = []
        chosen_indices = set()
        while len(new_dataset) < new_size:
            index = random.randint(0, len(dataset) - 1)
            if index not in chosen_indices:
                new_dataset.append(dataset[index])
                chosen_indices.add(index)

    resized_name = f"{key}_to_{new_size}_resized.json"
    save_dataset(new_dataset, resized_name)

    print(f"Resized Dataset saved to: {resized_name}")


def merge_datasets():
    print("Select the first dataset to merge:")
    first_dataset = select_dataset()
    key1, dataset1, traininginfo1 = load_dataset(first_dataset)

    print("Select the second dataset to merge:")
    second_dataset = select_dataset()
    key2, dataset2, traininginfo2 = load_dataset(second_dataset)

    print(f">> len1: {len(dataset1)} | len2 {len(dataset2)}")
    merged_dataset = dataset1 + dataset2
    print(f">> Total len: {len(merged_dataset)}")

    sort_key = gen_key()

    merged_name = f"{sort_key}_merged.json"
    save_dataset(merged_dataset, merged_name)

    print(f"Merged Dataset saved to: {merged_name}")


def merge_and_sort_datasets():
    print("Select datasets to merge and sort (select at least two):")
    datasets = []

    while True:
        dataset_path = select_dataset()
        if dataset_path:
            datasets.append(dataset_path)
            more = questionary.confirm(
                "Do you want to add another dataset?").ask()
            if not more:
                break
        else:
            break

    all_data = []
    keys = []

    for dataset in datasets:
        try:
            key, data, _ = load_dataset(dataset)
            all_data.extend(data)
            if isinstance(key, str):
                try:
                    key = int(key)
                except ValueError:
                    key = 0
            keys.append(key)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error loading {dataset}: {e}")
            continue

    print(f">> Total initial len: {len(all_data)}")
    input("Confirm")

    unique_data = []
    seen = set()
    sort_key = gen_key()
    for item in all_data:
        if has_valid_move(item[0]):
            item_tuple = tuple(map(tuple, item[0]))
            if item_tuple not in seen:
                seen.add(item_tuple)
                unique_data.append(item)

    print(f">> Total unique len: {len(unique_data)}")
    input("Confirm")

    merged_name = sort_key + "_sorted.json"
    save_dataset(unique_data, merged_name)

    print(f"Merged and Sorted Dataset saved to: {merged_name}")


def merge_and_sort_all_datasets_in_folder():
    datasets = [f for f in os.listdir("datasets/") if f.endswith(".json")]

    if len(datasets) < 2:
        print("Not enough datasets to merge and sort.")
        return

    all_data = []
    keys = []

    print(f">> Total files loaded: {len(datasets)}")

    for dataset in datasets:
        try:
            key, data, _ = load_dataset(dataset)
            all_data.extend(data)
            if isinstance(key, str):
                try:
                    key = int(key)
                except ValueError:
                    key = 0
            keys.append(key)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error loading {dataset}: {e}")
            continue

    print(f">> Total initial len: {len(all_data)}")
    input("Confirm length and start sorting!")

    unique_data = []
    seen = set()
    for item in all_data:
        if has_valid_move(item[0]):
            item_tuple = tuple(map(tuple, item[0]))
            if item_tuple not in seen:
                seen.add(item_tuple)
                unique_data.append(item)

    print(f">> Total unique len: {len(unique_data)}")

    sort_key = f"ALL_{gen_key()}"
    merged_name = f"{sort_key}_sorted.json"
    save_dataset(unique_data, merged_name)

    print(f"Merged and Sorted Dataset saved to: {merged_name}")


def predict_dataset():
    selected_dataset = select_dataset()
    dataset_data = load_dataset(selected_dataset)
    key, dataset, training_info = dataset_data

    new_dataset = []
    completed = 0
    data_len = len(dataset)
    print(f">> Loaded {len(dataset)} Moves. Start predicting")

    for move in dataset:
        board = move[0]
        print("- - - - - - - - - - - - - - - -")
        print(f">> | Board {completed} / {data_len}")
        print_board(board)
        predicted_action = monte_predict(board, 500, 10)
        print(predicted_action)
        new_dataset.append([board, predicted_action['predicted']])
        print("- - - - - - - - - - - - - - - -")
        completed += 1

    print(">> Completed Predicting")
    save_dataset(new_dataset, f"Predicted_{key}_training.json")


def main():

    while True:
        choice = questionary.select(
            "Choose an option:",
            choices=[
                "Show Dataset Size",
                "Play and Create Datasets",
                "Predict Dataset",
                "Merge Datasets",
                "Merge and Sort Datasets",
                "Merge and Sort all Datasets",
                "User finetune Dataset",
                "Split Dataset",
                "Augment Dataset",
                "Resize Dataset",
                "Shuffle Dataset",
                "Create Games Dataset",
                "Create Random Dataset",
                "Train existing Model with Dataset",
                "Train new Model with Dataset",
                "Exit",
            ],
        ).ask()

        if choice == "Create Games Dataset":
            num_games = questionary.text(
                "Number of games to generate:", default="1000"
            ).ask()
            if not num_games:
                print(f"No Imput, going back. . .")
                continue
            filename = questionary.text(
                "Filename to save the dataset:", default="gen_games.json"
            ).ask()
            if not filename:
                print(f"No Imput, going back. . .")
                continue
            create_games_dataset(int(num_games), filename=filename)

        elif choice == "Create Random Dataset":
            num_samples = questionary.text(
                "Number of samples to generate:", default="10000"
            ).ask()
            if not num_samples:
                print(f"No Imput, going back. . .")
                continue
            filename = questionary.text(
                "Filename to save the dataset:", default="gen_random.json"
            ).ask()
            if not filename:
                print(f"No Imput, going back. . .")
                continue
            create_random_dataset(int(num_samples), filename=filename)

        elif choice == "Train existing Model with Dataset":
            model_path = select_model("saves/")
            model = load_model(f"saves/{model_path}", compile=False)
            model.compile(
                loss="mean_squared_error", optimizer=Adam(learning_rate=0.005)
            )
            dataset_path = select_dataset()
            if not dataset_path:
                print(f"No Imput, going back. . .")
                continue
            dataset = load_dataset(dataset_path)
            num_epochs = questionary.text(
                "Number of epochs:", default="30").ask()
            batch_size = questionary.select(
                "Select Batch-Size (Default: 32):", ["32", "64", "128"]
            ).ask()
            print(num_epochs, batch_size)
            if not batch_size or not num_epochs:
                print("No no no, you have to select something.")
                input("Press any key to sell your soul.")
            train_model_with_dataset(
                model, dataset, int(num_epochs), int(batch_size))
            model.save(f"saves/{model_path}")
            print(f"Model trained and saved to saves/{model_path}")

        elif choice == "Train new Model with Dataset":
            model = create_model()
            dataset_path = select_dataset()
            if not dataset_path:
                print(f"No Imput, going back. . .")
                continue
            modelkey, dataset, training_info = load_dataset(dataset_path)
            num_epochs = questionary.text(
                "Number of epochs:", default="30").ask()
            batch_size = questionary.select(
                "Select Batch-Size (Default: 32):", ["32", "64", "128"]
            ).ask()
            if not batch_size or not num_epochs:
                print("No no no, you have to select something.")
                input("Press any key to sell your soul.")
            train_model_with_dataset(
                model, dataset, int(num_epochs), int(batch_size))
            model_key = gen_key()
            model.save(f"saves/trained/{model_key}_Model.h5")
            print(f"New model trained and saved to saves/{model_key}_Model.h5")

        elif choice == "Show Dataset Size":
            chosen_dataset = select_dataset()
            key, dataset, traininginfo = load_dataset(chosen_dataset)
            print(f">> Dataset: {chosen_dataset}")
            print(f">> Lengt: {len(dataset)}")
            input("Press ANY Key to continue. . .")

        elif choice == "Augment Dataset":
            chosen_dataset = select_dataset()
            if chosen_dataset:
                modelkey, aug_dataset, training_info = load_dataset(
                    chosen_dataset)
                augmented_dataset = generate_augmented_dataset(aug_dataset)
                new_augmented_dataset = []

                for turn in augmented_dataset:
                    if isinstance(turn[0], np.ndarray):
                        new_turn = (turn[0].tolist(),) + turn[1:]
                    else:
                        new_turn = (
                            [
                                row.tolist() if isinstance(row, np.ndarray) else row
                                for row in turn[0]
                            ],
                        ) + turn[1:]
                    new_augmented_dataset.append(new_turn)

                print(f">> Total len: {len(new_augmented_dataset)}")
                augmented_name = f"{modelkey}_augmented.json"
                save_dataset(new_augmented_dataset, augmented_name)

        elif choice == "User finetune Dataset":
            chosen_dataset = select_dataset()
            if chosen_dataset:
                modelkey, aug_dataset, training_info = load_dataset(
                    chosen_dataset)
                finetuned_dataset = user_finetune_augemntation(aug_dataset)
                print(f">> Total len: {len(finetuned_dataset)}")

                if questionary.confirm("Do you want to save this Model?"):
                    finetuned_name = f"{modelkey}_finetune_augmented.json"
                    save_dataset(finetuned_dataset, finetuned_name)

        elif choice == "Resize Dataset":
            resize_dataset()

        elif choice == "Merge Datasets":
            merge_datasets()

        elif choice == "Merge and Sort Datasets":
            merge_and_sort_datasets()

        elif choice == "Merge and Sort all Datasets":
            merge_and_sort_all_datasets_in_folder()

        elif choice == "Create Dataset while playing":
            game_console()

        elif choice == "Shuffle Dataset":
            create_shuffled_copy()

        elif choice == "Split Dataset":
            split_dataset()

        elif choice == "Predict Dataset":
            predict_dataset()

        elif choice == "Exit" or not choice:
            return


if __name__ == "__main__":
    main()
