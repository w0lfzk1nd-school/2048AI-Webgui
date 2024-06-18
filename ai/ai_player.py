import os
import random
import numpy as np
from time import sleep
from ai.environment import Environment
import json
from utils.utilities import print_and_log


def save_to_dataset(data):
    with open("dataset.json", "a") as f:
        json.dump(data, f)
        f.write("\n")
    print_and_log("Written step into Dataset")


class AIPlayer:
    def __init__(self, model=None, epsilon=1.0, epsilon_min=0.07, epsilon_decay=0.994):
        self.model = model
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = []
        self.gamma = 0.95
        self.last_actions = [None, None]

    def reset_epsi(self):
        self.epsilon = 1.0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        # os.system("cls")
        if np.random.rand() <= self.epsilon or self.is_last_actions_same(state):
            print_and_log(
                f"==================================================\n>> Random Action (Epsilon: {self.epsilon})"
            )
            action = random.randrange(4)
        else:
            act_values = self.model.predict(state.reshape(1, 4, 4))
            action = np.argmax(act_values[0])
            print_and_log(
                f"==================================================\n>> Predicted Action Values: {act_values} (Epsilon: {self.epsilon})"
            )
        self.last_actions.append(action)
        return action

    def is_last_actions_same(self, state):
        if self.last_actions[0] is not None and self.last_actions[1] is not None:
            prev_act_values = self.model.predict(state.reshape(1, 4, 4))
            return np.array_equal(
                prev_act_values, self.last_actions[0]
            ) or np.array_equal(prev_act_values, self.last_actions[1])
        return False

    def replay(self, batch_size=32):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.max(
                    self.model.predict(next_state.reshape(1, 4, 4))[0]
                )
            target_f = self.model.predict(state.reshape(1, 4, 4))
            target_f[0][action] = target
            self.model.train_on_batch(state.reshape(1, 4, 4), target_f)

    def load(self, name):
        from tensorflow.keras.models import load_model

        self.model = load_model(name)

    def ai_play_game(self, num_games=1):
        from ai.train import setup_training
        from ai.ai_player import AIPlayer as ai_player

        env = Environment()
        model, json_path, training_info = setup_training()
        scores = {"high": 0, "best": 0}
        try:
            for _ in range(num_games):
                # ai_player.reset_epsi()
                state = env.reset()
                done = False
                old_state = None
                steps = 0
                while not done:
                    steps += 1
                    if steps > 500:
                        print_and_log("Game finished at 500 moves")
                    # os.system("cls")
                    old_state = state
                    if env.game.highscore > scores["high"]:
                        scores["high"] = env.game.highscore

                    if env.game.bestblock > scores["best"]:
                        scores["best"] = env.game.bestblock

                    action = self.choose_action(state)
                    state, _, done = env.step(action)
                    env.render(scores, training_info["traintime"])
                    choice = ["UP", "LEFT", "DOWN", "RIGHT"][action]
                    print(f">> {choice}")
                    sleep(2)
                print_and_log("Spiel beendet. Endstand:", env.game.score)
        except KeyboardInterrupt:
            return
        except Exception as e:
            print(f"Error: {e}")
            input("\n\nConfirm error")
            return