import logging
import os
import random
from datetime import datetime
import json
import questionary
from logging.handlers import RotatingFileHandler


def setup_logging(file="main", max_size_mb=50, backup_count=5):
    try:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logs_dir = "logs"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
            print(f"Log directory created at {logs_dir}")
        log_filename = f"logs/log-{file}_{current_time}.txt"
        print(f"Log file will be created at {log_filename}")

        handler = RotatingFileHandler(
            filename=log_filename,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s: %(message)s"))
        logging.getLogger().addHandler(handler)
        
        logging.info("Logging setup complete.")
    except Exception as e:
        print(f"Failed to set up logging: {str(e)}")
        exit(1)


def print_and_log(message="", log=True):
    """Prints a message and logs it if needed."""
    print(message)
    if log:
        hi = "hi"
        logging.info(message)


def log(message=""):
    """Lofs a message."""
    logging.info(message)


def select_model(path="saves/"):
    """Zeigt ein Auswahlmenü für verfügbare Modelldateien."""
    files = [f for f in os.listdir(path) if f.endswith(".h5")]
    if files != []:
        return questionary.select("Choose a 'model' to load:", choices=files).ask()
    else:
        return False


def select_dataset(path="datasets/"):
    """Zeigt ein Auswahlmenü für verfügbare Modelldateien."""
    files = [
        f
        for f in os.listdir(path)
        if f.endswith(
            (
                "hooman_game.json",
                "split.json",
                "random_boards.json",
                "gen_random.json",
                "gen_games.json",
                "resized.json",
                "merged.json",
                "web.json",
                "gen.json",
                "training.json",
                "shuffled.json",
                "augmented.json",
                "sorted.json",
                "moves.json",
                "dataset.json"
            )
        )
    ]
    if not files:
        print("Keine passenden Dateien im angegebenen Verzeichnis gefunden.")
        return None
    return questionary.select("Choose a 'dataset' to load:", choices=files).ask()


def save_training_info(info, path):
    """Speichert Trainingsinformationen in einer JSON-Datei."""
    with open(path, "w") as f:
        json.dump(info, f, indent=4)


def get_current_time():
    """Gibt das aktuelle Datum und Uhrzeit formatiert zurück."""
    return datetime.now().strftime("%H-%M-%S_%d-%m-%Y")

def get_cute_time():
    """Gibt das aktuelle Datum und Uhrzeit schön formatiert zurück."""
    return datetime.now().strftime("%H:%M:%S %d/%m/%Y")

def is_coma(x):
    """Prüft, ob eine Zahl eine Kommazahl ist."""
    return int(x) != x


def get_clean_filename(file_path):
    """Nimmt Pfade zu JSON Dateien und filtert den rohen Dateinamen aus."""
    filename = os.path.basename(file_path)
    clean_name = filename.replace("-data.json", "")
    return clean_name


def get_clean_key(file_path):
    """Nimmt Pfade zu JSON Dateien und filtert den rohen Dateinamen aus."""
    filename = os.path.basename(file_path)
    clean_name = filename
    remove_parts = [
        "_hooman_game.json",
        "_random_boards.json",
        "_split.json",
        "_gen_games.json",
        "_gen_random.json",
        "_merged.json",
        "_resized.json",
        "_shuffled.json" "-data.json",
        "_Model.h5",
        "_augmented.json",
        "_finetune_augmented.json",
        "_web.json",
        "_gen.json",
        "_dataset.json",
        ".-data",
        "_moves.json",
        "_training.json",
        ".json",
    ]
    for part in remove_parts:
        clean_name = clean_name.replace(part, "")
    return clean_name


def gen_key(val="gen"):
    characters = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "0",
    ]
    random_key = "".join(random.choice(characters) for _ in range(6))
    model_key = f"{random_key}"
    return model_key


def form_time(ms):
    hours = ms // (1000 * 60 * 60)
    remaining_ms = ms % (1000 * 60 * 60)
    minutes = remaining_ms // (1000 * 60)
    remaining_ms %= 1000 * 60
    seconds = remaining_ms // 1000
    milliseconds = remaining_ms % 1000

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s {round(milliseconds)}ms"
    elif minutes > 0:
        return f"{minutes}m {seconds}s {round(milliseconds)}ms"
    elif seconds > 0:
        return f"{seconds}s {round(milliseconds)}ms"
    else:
        return f"{round(milliseconds)}ms"
