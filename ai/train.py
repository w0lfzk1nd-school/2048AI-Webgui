from ai.model import create_model
from ai.environment import Environment
from ai.ai_player import AIPlayer, save_to_dataset
import numpy as np
import random
import questionary
import datetime
import json
import os
from keras.optimizers import Adam  # Importieren Sie den Optimierer
from utils.utilities import (
    print_and_log,
    select_model,
    gen_key,
    form_time,
    get_clean_key,
    log,
    get_current_time
)
from tensorflow.keras.models import load_model

#def train_from_replay(ai_player, batch):
#    print(">> Training from memory!")
#    for state, action, reward, next_state, done in batch:
#        state = np.reshape(state, (1, 4, 4, 1))
#        next_state = np.reshape(next_state, (1, 4, 4, 1))
#        target = reward
#        if not done:
#            target += ai_player.gamma * np.amax(ai_player.model.predict(next_state)[0])
#        target_f = ai_player.model.predict(state)
#        target_f[0][action] = target
#        ai_player.model.fit(state, target_f, epochs=1, verbose=0)

def train_model(episodes=1000, mode="interactive"):
    global highscore, bestscore
    model, json_path, training_info = setup_training()
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')

    env = Environment()
    env.model = f"{training_info['model_name']}"
    ai_player = AIPlayer(model)
    exit = False
    highscore = 0
    bestscore = 0
    train_start = datetime.datetime.now()

    for episode in range(int(episodes)):
        #ai_player.reset_epsi()
        print_and_log(f"Episode: {episode}/{episodes}")
        if env.game.highscore > highscore:
            highscore = env.game.highscore
            env.game.tot_highscore = highscore

        if env.game.bestblock > bestscore:
            bestscore = env.game.bestblock

        state = env.reset()
        state = np.reshape(state, (1, 4, 4, 1))
        done = False
        steps = 0
        if exit:
            break

        while not done and not exit:
            try:
                env.render(
                    {"high": highscore, "best": bestscore}, training_info["traintime"]
                )
                if mode == "interactive":
                    user_input = input(
                        "Dein Zug (W: oben, A: links, S: unten, D: rechts) oder 'exit' zum Beenden: "
                    ).lower()
                    if user_input.lower() == "exit":
                        if save_training(model, training_info, json_path):
                            return
                        else:
                            continue
                    action = get_valid_action(user_input)
                    if action is None:
                        continue
                else:
                    action = ai_player.choose_action(state)

                next_state, reward, done = env.step(action)
                next_state = np.reshape(next_state, (1, 4, 4, 1))
                ai_player.remember(state, action, reward, next_state, done)
                
                # Training des Modells mit einem zufälligen Batch aus dem Speicher
                if len(ai_player.memory) % ai_player.batch_size*4 == 0:
                    ai_player.replay()

                if len(ai_player.memory) % 30 == 0:
                    current_time = datetime.datetime.now()
                    time_trained = (current_time - train_start).total_seconds() * 1000
                    training_info["traintime"] += round(time_trained, 3)
                    train_start = current_time

                state = next_state
                steps += 1
                if steps > 500:
                    if env.game.bestblock > 2048:
                        training_info["total_win"] += 1
                        print(
                            "# # # # # # # # # # # # # # \nGame finished with WIN at 500 moves.\n# # # # # # # # # # # # # #"
                        )
                        log(
                            "# # # # # # # # # # # # # # \nGame finished with WIN at 500 moves.\n# # # # # # # # # # # # # #"
                        )
                        break
                    else:
                        print(
                            "# # # # # # # # # # # # # # \nGame finished NO WIN at 500 moves.\n# # # # # # # # # # # # # #"
                        )
                        log(
                            "# # # # # # # # # # # # # # \nGame finished NO WIN at 500 moves.\n# # # # # # # # # # # # # #"
                        )
                        break
            except KeyboardInterrupt:
                exit = True

        ai_player.epsilon *= ai_player.epsilon_decay
        ai_player.epsilon = max(ai_player.epsilon_min, ai_player.epsilon)
        ai_player.replay()

        env.runtime_history.append(round(time_trained, 3))
        print_and_log(
            f"Zeit für diese Episode: {round(time_trained, 3)} ms, Gesamttrainingszeit: {training_info['traintime']} ms"
        )
        update_training_info(training_info, steps, env.game.score)
        train_start = datetime.datetime.now()

    finalize_training(model, training_info, json_path, env,  ai_player.memory)


def setup_training():
    load_existing = (
        input("Möchten Sie ein existierendes Modell laden? (y/n): ").lower() == "y"
    )
    if load_existing:
        model_path = select_model("saves/")
        model = load_model(f"saves/{model_path}")
        json_path = f"saves/{model_path.replace('h5', '-data.json')}"
        print(json_path)
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                training_info = json.load(f)
        else:
            training_info = init_training_info(json_path)
    else:
        model_key = gen_key()
        model = create_model()
        json_path = f"saves/{model_key}-data.json"
        training_info = init_training_info(json_path)
    return model, json_path, training_info


def init_training_info(json_path):
    training_info = {
        "model_name": get_clean_key(json_path.split("/")[-1]),
        "model_start": datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S"),
        "last_training": "",
        "traintime": 0,
        "total_train_steps": 0,
        "total_games": 0,
        "total_win": 0,
        "avg_points": 0,
        "scores": {"highscore": 0, "bestblock": 0},
    }
    with open(json_path, "w") as f:
        json.dump(training_info, f)
    return training_info


def update_training_info(training_info, steps, score):
    training_info["total_train_steps"] += steps
    training_info["total_games"] += 1
    training_info["avg_points"] = (
        training_info["avg_points"] * (training_info["total_games"] - 1) + score
    ) / training_info["total_games"]
    if training_info["scores"]["highscore"] < highscore:
        training_info["highscore"] = highscore
    elif training_info["scores"]["bestblock"] < bestscore:
        training_info["bestblock"] = bestscore


def save_training(model, training_info, memory, json_path):
    json_path = f"saves/trained/{training_info['model_name']}_trained.h5"
    model.save(json_path)
    training_info["last_training"] = datetime.datetime.now().strftime(
        "%Y.%m.%d %H:%M:%S"
    )
    with open(json_path, "w") as f:
        json.dump(training_info, f)

    save_key = training_info["model_name"]

    with open(json_path, "w") as f:
        json.dump(training_info, f)

    print(f">> Trainingdata saved: {save_key}_training.json")

    #with open(f"datasets/trainings/{save_key}-{get_current_time()}_training.json", "w") as f:
    #    json.dump(memory.tolist(), f)
    #print(">> Dataset saved")

    print_and_log("Training finished and saved.")
    return True


def finalize_training(model, training_info, json_path, env, memory):        
    save_training(model, training_info, memory, json_path)
    env.plot_history()
    print_and_log("Training finished.")


def get_valid_action(user_input):
    # Input mapping
    direction_map = {"w": 0, "a": 1, "s": 2, "d": 3}
    action = direction_map.get(user_input)
    if action is None:
        print("Invalid input. Inputs are: 'W', 'A', 'S' and 'D'.")
    return action


if __name__ == "__main__":
    train_model(episodes=500, mode="interactive")
