import numpy as np
import json
import random
import os
import multiprocessing
from multiprocessing import Process, Queue
import questionary
from ai.environment import Environment
from ai.train import init_training_info
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from utils.utilities import select_model, select_dataset, gen_key, get_clean_key
import math


def create_games_dataset(num_games=1000, max_moves=500, filename="games_dataset.json"):
    num_processes = 4
    games_per_process = math.ceil(num_games // num_processes)
    queues = []

    for _ in range(num_processes):
        queue = Queue()
        process = Process(
            target=generate_game_data, args=(games_per_process, max_moves, queue)
        )
        queues.append(queue)
        process.start()

    combined_dataset = []
    for queue in queues:
        combined_dataset.extend(queue.get())

    save_dataset(combined_dataset, filename)
    print(
        f"Games dataset generated with {len(combined_dataset)} samples and saved to {filename}"
    )


def generate_game_data(num_games, max_moves, queue):
    dataset = []
    env = Environment()

    for _ in range(num_games):
        state = env.reset()
        game_data = []
        done = False
        move_count = 0

        while not done and move_count < max_moves:
            action = random.randrange(4)
            next_state, reward, done = env.step(action)
            if isinstance(state, np.ndarray):
                board_list = state.tolist()
            else:
                board_list = [
                    row.tolist() if isinstance(row, np.ndarray) else row
                    for row in state
                ]

            game_data.append((board_list, action))
            state = next_state
            move_count += 1

        dataset.extend(game_data)

    queue.put(dataset)


def create_random_dataset(num_samples=10000, filename="random_dataset.json"):
    num_processes = 4
    samples_per_process = math.ceil(num_samples // num_processes)
    queues = []

    for _ in range(num_processes):
        queue = Queue()
        process = Process(
            target=generate_random_data, args=(samples_per_process, queue)
        )
        queues.append(queue)
        process.start()

    combined_dataset = []
    for queue in queues:
        combined_dataset.extend(queue.get())

    save_dataset(combined_dataset, filename)
    print(
        f"Random dataset generated with {len(combined_dataset)} samples and saved to {filename}"
    )


def generate_random_data(num_samples, queue):
    dataset = []

    for _ in range(num_samples):
        state = generate_valid_board()
        action = random.randrange(4)  # Random action
        if isinstance(state, np.ndarray):
            board_list = state.tolist()
        else:
            board_list = [
                row.tolist() if isinstance(row, np.ndarray) else row
                for row in state
            ]
        dataset.append((board_list, action))
        dataset.append((state, action))

    queue.put(dataset)


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
                if (
                    random.random() < 0.3
                ):
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
                board[i][j] = random.choice([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])

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
        json.dump(dataset, f)


def load_dataset(filename):
    cleankey = get_clean_key(filename)
    with open(f"datasets/{filename}", "r") as f:
        dataset = json.load(f)
    if os.path.exists(f"datasets/{cleankey}-data.json"):
        with open(f"datasets/{cleankey}-data.json", "r") as f:
            training_info = json.load(f)
    else:
        training_info = init_training_info(f"datasets/{cleankey}-data.json")
    return cleankey, dataset, training_info


def preprocess_dataset(dataset):
    states = np.array([np.array(data[0]) for data in dataset])
    actions = np.array([data[1] for data in dataset])
    return states, actions


def create_model(input_shape):
    model = Sequential(
        [
            Flatten(input_shape=input_shape),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(4, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model_with_dataset(model, dataset, epochs=10, batch_size=32):
    key, dataset_data, training = dataset
    print(f"Dataset len: {len(dataset_data)}")
    states, actions = preprocess_dataset(dataset_data)

    # One-hot encode actions
    actions_one_hot = np.zeros((actions.size, 4))
    actions_one_hot[np.arange(actions.size), actions] = 1

    # Train the model
    model.fit(states, actions_one_hot, epochs=epochs, batch_size=batch_size)


def parallel_training(model_path, dataset_path, epochs):
    model = load_model(f"saves/{model_path}", compile=False)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    dataset = load_dataset(dataset_path)
    train_model_with_dataset(model, dataset, epochs)
    model.save(f"saves/{model_path}")


def main():
    import questionary

    while True:
        choice = questionary.select(
            "Choose an option:",
            choices=[
                "Create Games Dataset",
                "Create Random Dataset",
                "Train existing Model with Dataset",
                "Train new Model with Dataset",
                "Train Model with Multiple Processes",
                "Exit",
            ],
        ).ask()

        if choice == "Create Games Dataset":
            num_games = questionary.text(
                "Number of games to generate:", default="10"
            ).ask()
            if not num_games:
                print(f"No Input, going back. . .")
                continue
            filename = questionary.text(
                "Filename to save the dataset:", default="gen_games.json"
            ).ask()
            if not filename:
                print(f"No Input, going back. . .")
                continue
            create_games_dataset(int(num_games), filename=filename)

        elif choice == "Create Random Dataset":
            num_samples = questionary.text(
                "Number of samples to generate:", default="10"
            ).ask()
            if not num_samples:
                print(f"No Input, going back. . .")
                continue
            filename = questionary.text(
                "Filename to save the dataset:", default="gen_random.json"
            ).ask()
            if not filename:
                print(f"No Input, going back. . .")
                continue
            create_random_dataset(int(num_samples), filename=filename)

        elif choice == "Train existing Model with Dataset":
            model_path = select_model("saves/")
            model = load_model(f"saves/{model_path}", compile=False)
            model.compile(
                optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
            )
            dataset_path = select_dataset()
            if not dataset_path:
                print(f"No Input, going back. . .")
                continue
            dataset = load_dataset(dataset_path)
            num_epochs = questionary.text("Number of epochs:", default="15").ask()
            train_model_with_dataset(model, dataset, int(num_epochs))
            model.save(f"saves/{model_path}")
            print(f"Model trained and saved to saves/{model_path}")

        elif choice == "Train new Model with Dataset":
            model = create_model((4, 4))
            dataset_path = select_dataset()
            if not dataset_path:
                print(f"No Input, going back. . .")
                continue
            dataset = load_dataset(dataset_path)
            num_epochs = questionary.text("Number of epochs:", default="15").ask()
            train_model_with_dataset(model, dataset, int(num_epochs))
            model_key = gen_key()
            model.save(f"saves/trained/{model_key}_Model.h5")
            print(f"New model trained and saved to saves/{model_key}_Model.h5")

        elif choice == "Train Model with Multiple Processes":
            model_path = select_model("saves/")
            dataset_paths = [select_dataset() for _ in range(4)]
            if any(not dataset_path for dataset_path in dataset_paths):
                print(f"No Input, going back. . .")
                continue
            num_epochs = questionary.text("Number of epochs:", default="1000").ask()

            processes = []
            for dataset_path in dataset_paths:
                p = multiprocessing.Process(
                    target=parallel_training,
                    args=(model_path, dataset_path, int(num_epochs)),
                )
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            print(
                f"Model trained with multiple datasets and saved to saves/{model_path}"
            )

        elif choice == "Exit":
            break


if __name__ == "__main__":
    main()
