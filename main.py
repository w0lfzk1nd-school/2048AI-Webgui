import os
import json
import questionary
from game.game_console import main as play_game
from ai.train import train_model
from ai.model import check_accuracy
from utils.utilities import select_model, setup_logging
from ai.ai_player import AIPlayer
from ai.dataset import main as dataset_main
from ai.dataset import load_dataset
from ai.dataset_multi import main as multi_dataset_main
from tensorflow.keras.models import load_model

setup_logging()

def load_and_play():
    model_path = select_model("saves/")
    if not model_path:
        print("Kein Modell vorhanden oder ausgewählt.")
        return

    # Lade Modell-Details
    json_path = model_path.replace(".h5", "-data.json")
    try:
        with open(os.path.join("saves", json_path), "r") as json_file:
            model_details = json.load(json_file)
            print("Modell-Details:")
            for key, value in model_details.items():
                print(f"{key}: {value}")
    except FileNotFoundError:
        print("Keine Detaildatei gefunden für das ausgewählte Modell.")

    input("Drücken Sie eine beliebige Taste, um das Modell spielen zu lassen...")

    # Modell spielen lassen
    player = AIPlayer()
    player.load(os.path.join("saves", model_path))
    player.ai_play_game(num_games=5)


def main_menu():
    while True:
        os.system("cls")
        print("\n\n\n")
        response = questionary.select(
            "Was möchten Sie tun?",
            choices=[
                "Spiel selbst spielen",
                "Datasets",
                "AI selbst trainieren",
                "AI trainieren",
                "AI spielen lassen",
                "AI Model Accuracy",
                "Exit",
            ],
        ).ask()

        if response == "Spiel selbst spielen":
            play_game()
        elif response == "AI trainieren":
            games = questionary.text("Anzahl der Runden zum spielen:").ask()
            if isinstance(int(games), int):
                train_model(episodes=games, mode="auto")
        elif response == "AI selbst trainieren":
            games = questionary.text("Anzahl der Runden zum spielen:").ask()
            if isinstance(int(games), int):
                train_model(episodes=games, mode="interactive")
        elif response == "AI spielen lassen":
            load_and_play()
        elif response == "Datasets":
            dataset_main()
        elif response == "Multi-Process Datasets":
            multi_dataset_main()
        elif response == "AI Model Accuracy":
            sel_model = select_model()
            model = load_model(f"saves/{sel_model}")
            if not model:
                input(">> No Model selected!")
            key, validation_dataset, traininginfo = load_dataset("validation_data.json")
            check_accuracy(model, validation_dataset)
        elif not response or response == "Exit":
            print("Danke fürs Spielen!")
            break


if __name__ == "__main__":
    main_menu()
