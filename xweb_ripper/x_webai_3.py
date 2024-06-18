from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By as by
import time
import numpy as np
from bs4 import BeautifulSoup
import numpy as np
from game.game_console import print_board
from ai.dataset import save_dataset
from utils.utilities import gen_key
import json

# Setze den Dateipfad für die JSON-Datei
key = gen_key()
json_file_paths = f"datasets/web/{key}_web.json"
bar = "- - - - - - - - - - - - - - - - - - - - - - - -"


# Funktion zum Speichern der Züge in einer JSON-Datei
def save_moves(moves):
    key = gen_key()
    json_file_path = f"datasets/web/{key}_web.json"
    with open(json_file_path, "w") as f:
        json.dump(moves, f)


# Variable zum Speichern des vorherigen Spielfelds
prev_board = np.zeros((4, 4), dtype=int)
# Array zum Speichern der Züge
moves = []

## Setze die Optionen für den Firefox-Browser
options = Options()
options.add_argument('--headless')  # Optional: Führe den Browser im Hintergrund aus
options.headless = False
gecko_driver_path = "geckodriver.exe"
firefox_binary_path = "C:/Program Files/Mozilla Firefox/firefox.exe"
service = Service(gecko_driver_path)
options.binary_location = firefox_binary_path
driver = webdriver.Firefox(service=service, options=options)
time.sleep(10)
# Navigiere zum Spiel 2048
driver.get("https://sleepycoder.github.io/2048/")

# XPath für das Spielfeld
grid_xpath = "/html/body/div/div[3]"


def get_tile_values(driver):
    # Warte, bis das Spielfeld geladen ist
    time.sleep(2)
    # HTML-Code des Spielfelds abrufen
    grid_html = driver.find_element(by.XPATH, grid_xpath).get_attribute("outerHTML")
    # Parse das HTML mit BeautifulSoup
    soup = BeautifulSoup(grid_html, "html.parser")
    # print(soup)
    # Extrahiere das Spielfeld
    board = np.zeros((4, 4), dtype=int)
    tiles_with_values = soup.find(class_="tile-container").find_all(class_="tile")
    for tile in tiles_with_values:
        position = tile["class"][2].split("-")[-2:]
        col, row = map(int, position)
        value = int(tile.find(class_="tile-inner").text)
        board[row - 1, col - 1] = value
    return board


def print_board(board):
    for row in board:
        print(row)
    print()


import numpy as np


def slide_left(board):
    new_board = []
    for row in board:
        new_row = [num for num in row if num != 0]
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1]:
                new_row[i] *= 2
                new_row[i + 1] = 0
        new_row = [num for num in new_row if num != 0]
        new_board.append(new_row + [0] * (len(row) - len(new_row)))
    return np.array(new_board)


def slide_right(board):
    return np.fliplr(slide_left(np.fliplr(board)))


def slide_up(board):
    return np.rot90(slide_left(np.rot90(board, 1)), -1)


def slide_down(board):
    return np.rot90(slide_left(np.rot90(board, -1)), 1)


def detect_move_direction(old_board, current_board):
    old_board = np.array(old_board)
    current_board = np.array(current_board)

    # Check all directions
    for direction, slide_func in enumerate(
        [slide_up, slide_left, slide_down, slide_right]
    ):
        simulated_board = slide_func(old_board)
        temp_board = np.copy(current_board)

        # print()
        #
        # print("Current")
        # print_board(current_board)
        #
        # print("Before check")
        # print_board(temp_board)

        # Remove predicted values from temp_board
        for row in range(simulated_board.shape[0]):
            for col in range(simulated_board.shape[1]):
                if simulated_board[row, col] == current_board[row, col]:
                    temp_board[row, col] = 0

        # Check if the last remaining non-zero value is 2 or 4
        non_zero_values = temp_board[temp_board != 0]
        if len(non_zero_values) == 1 and (
            non_zero_values[0] == 2 or non_zero_values[0] == 4
        ):
            return direction

    return -1

games = 75
done_games = 0

step_button = "/html/body/div/div[2]/div/button[3]"
try_again = "/html/body/div/div[3]/div[1]/div/a[2]"

try:
    while done_games < games:
        current_board = get_tile_values(driver)
        total_value = sum(sum(row) for row in current_board)
        
        if total_value == 0:
            print("Copy Board")
            prev_board = current_board.copy()
            continue

        if (
            prev_board is not None
            and not np.array_equal(current_board, prev_board)
        ):
            print("-" * 40)
            print(f">> Game {done_games}/{games}\n")
            print_board(current_board)
            print("-" * 40)

            prev_board = np.array(prev_board)
            current_board = np.array(current_board)

            action = detect_move_direction(prev_board, current_board)
            if action != -1:
                moves.append([current_board.tolist(), action])
                form_action = ["Up", "Left", "Down", "Right"][action]
                print(f">> Last Action Taken: {action} | {form_action}\n{'-' * 40}")
            
            prev_board = current_board.copy()
        
        try:
            retry_element = driver.find_element(by.XPATH, try_again)
            if retry_element.is_displayed() and retry_element.is_enabled():
                retry_element.click()
                done_games += 1
                print(f"Game {done_games}/{games} completed.")
                prev_board =  np.zeros((4, 4), dtype=int)
                if done_games % 100 == 0 and moves:
                    save_moves(moves)
                    moves = []
            else:
                click_element = driver.find_element(by.XPATH, step_button)
                click_element.click()
        except Exception as e:
            print(f"Error: {e}")

except KeyboardInterrupt:
    pass
except Exception as e:
    print(f"Error: {e}")

finally:
    # Speichere die Bewegungen, falls noch nicht gespeichert
    if moves:
        save_moves(moves)

    # Schließe den Browser
    driver.quit()