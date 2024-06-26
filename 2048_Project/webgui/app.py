from flask import Flask, jsonify, request, render_template, session
from game_engine import Game2048
from multiprocessing import Process, Queue
from time import sleep
import numpy as np
import uuid
import os
import sys
from datetime import datetime
from threading import Lock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.utilities import (
    setup_logging,
    print_and_log,
    log,
    get_current_time,
    get_cute_time,
)
from ai.monte_carlo import monte_predict
import json
from utils.handle_db import DatabaseHandler

global move_file

app = Flask(__name__)

setup_logging("flask")

# Erstellen einer Instanz der DatabaseHandler-Klasse
db_handler = DatabaseHandler()

webgui_db = db_handler.handle_db('select', 'webgui')
webgui_leaderboard = db_handler.handle_db('select', 'web_leaderboard')

app.secret_key = os.urandom(24)
moves = []
move_file = ""
helpers = 0
turns = 0

# Define global variables and a lock
total_highscore = webgui_db[0][1]
total_bestblock = webgui_db[0][3]
total_highscore_txt = webgui_db[0][2]
total_bestblock_txt = webgui_db[0][4]
highscore_lock = Lock()

# Helper functions for serializing and deserializing the game state
def serialize_game(game):
    return {
        "board": game.board,
        "score": game.score,
        "highscore": game.highscore,
        "bestblock": game.bestblock,
    }


def deserialize_game(data):
    game = Game2048()
    game.board = data["board"]
    game.score = data["score"]
    game.highscore = data["highscore"]
    game.bestblock = data["bestblock"]
    return game


def get_game():
    if "game" in session:
        return deserialize_game(session["game"])
    else:
        game = Game2048()
        session["game"] = serialize_game(game)
        return game


def get_highscore():
    return {"high": total_highscore, "block": total_bestblock}


def get_webgui_high():
    return db_handler.handle_db('select', 'webgui')

def get_leaderboard():
    return db_handler.handle_db('select', 'web_leaderboard', None, "ORDER BY score DESC")

def add_move_to_json(file_path, board, action):
    try:
        with open(file_path, "r") as file:
            try:
                moves = json.load(file)
            except json.JSONDecodeError:
                moves = []
    except FileNotFoundError:
        moves = []

    if isinstance(board, np.ndarray):
        board_list = board.tolist()
    else:
        board_list = [
            row.tolist() if isinstance(row, np.ndarray) else row for row in board
        ]
    moves.append([board_list, action])
    with open(file_path, "w") as file:
        json.dump(moves, file)


def monte_predict_wrapper(board, steps, output_queue):
    result = monte_predict(board, steps)
    output_queue.put(result)


# === App Routes


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/get_highscore", methods=["GET"])
def handle_leaderboard():
    leaderboard_data = get_webgui_high()
    return jsonify(leaderboard_data[0])

@app.route("/api/get_leaderboard", methods=["GET"])
def get_leader():
    leaderboard_data = get_leaderboard()
    return jsonify(leaderboard_data)

@app.route('/api/add_highscore', methods=['POST'])
def add_highscore():
    data = request.get_json()
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    uname = data.get('uname')
    score = data.get('score')
    block = data.get('block')
    try:
        db_handler.handle_db('insert', 'web_leaderboard', {"time": current_time, "uname": uname, "score": score, "block": block})
        return jsonify({"status": "success"})
    except Exception as e:
        print(f"Leaderbaord Error: {e}")
        return jsonify({"status": "failed"})

@app.route('/api/update_highscore', methods=['POST'])
def update_highscore():
    data = request.get_json()
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    uname = data.get('uname')
    score = data.get('score')
    block = data.get('block')
    try:
        db_handler.handle_db('update', 'web_leaderboard', data={"time": current_time, "uname": uname, "score": score, "block": block}, condition=f"WHERE uname = {uname}")
        return jsonify({"status": "success"})
    except Exception as e:
        print(f"Leaderbaord Error: {e}")
        return jsonify({"status": "failed"})

@app.route("/api/generate_key", methods=["POST"])
def generate_key():
    global move_file
    data = request.get_json()
    if "key" in data:
        session_key = data["key"]
        session["key-value"] = session_key
    else:
        session_key = str(uuid.uuid4())
        session["key-value"] = session_key
    session["game"] = serialize_game(Game2048())
    print_and_log(
        f"- - - - - - - - - - - - - - - - -\n>> New Session @{get_current_time()}\n-- Key: {session_key}\n- - - - - - - - - - - - - - - - -"
    )
    move_file = f"datasets/{session_key}_moves.json"
    return jsonify({"key": session_key})


@app.route("/api/board", methods=["GET"])
def get_board():
    game = get_game()
    return jsonify(game.board)

# Game get highscores
@app.route("/api/score", methods=["GET"])
def get_cur_score():
    global total_highscore, total_bestblock, total_highscore_txt, total_bestblock_txt
    game = get_game()

    return jsonify(
        {
            "score": game.score,
            "block": game.bestblock,
            "highscore": total_highscore_txt,
            "best_block": total_bestblock_txt,
        }
    )

# Player reset game
@app.route("/api/reset", methods=["POST"])
def reset():
    session["game"] = serialize_game(Game2048())
    return "Game resetted"

# Player predict next move
@app.route("/api/predict", methods=["GET"])
def get_prediction():
    game = get_game()
    print(">> Init Monte Carlo Algorithm. . .")
    sleep(3)
    if game.bestblock > 512:
        steps = 200
    else:
        steps = 400

    output_queue = Queue()
    p = Process(target=monte_predict_wrapper, args=(game.board, steps, output_queue))
    p.start()
    p.join()
    
    monte_predicted = output_queue.get()

    if isinstance(monte_predicted["scores"], np.ndarray):
        monte_predicted["scores"] = monte_predicted["scores"].astype(str).tolist()

    monte_action = ["Up/Hoch", "Left/Links", "Down/Runter", "Right/Rechts"][
        monte_predicted["predicted"]
    ]
    monte_predicted["predicted_txt"] = monte_action
    monte_predicted["predicted"] = str(monte_predicted["predicted"])

    return jsonify(monte_predicted)

# Player moves
@app.route("/api/move/<int:direction>", methods=["POST"])
def make_move(direction):
    global move_file, moves, total_highscore, total_bestblock, total_highscore_txt, total_bestblock_txt
    game = get_game()
    old_board = session["game"]["board"]
    game_over = game.move(direction)
    session["game"] = serialize_game(game)
    current_board = game.board
    if game.score > total_highscore:
        total_highscore = game.score
        total_highscore_txt = f"[ {game.score} ] @{get_cute_time()}"
        db_handler.handle_db("update", "webgui", {'highscore_txt': total_highscore_txt})
        db_handler.handle_db("update", "webgui", {'highscore': game.score})

    if game.bestblock > total_bestblock:
        total_bestblock = game.bestblock
        total_bestblock_txt = f"[ {game.bestblock} ] @{get_cute_time()}"
        db_handler.handle_db("update", "webgui", {'highblock_txt': total_highscore_txt})
        db_handler.handle_db("update", "webgui", {'highblock': game.bestblock})
        
    if isinstance(current_board, np.ndarray):
        board_list = current_board.tolist()
    else:
        board_list = [
            row.tolist() if isinstance(row, np.ndarray) else row
            for row in current_board
        ]
    if not np.array_equal(old_board, game.board):
        moves.append([board_list, direction])

        if move_file == "":
            move_file = f"datasets/side_session_moves.json"

        add_move_to_json(move_file, board_list, direction)

        print_and_log(f">> ({get_current_time()}) Finished turn and saved turn.")
        return jsonify({"board": board_list, "game_over": game_over, "score": str(game.score), "block": str(game.bestblock)})


@app.route("/api/moves", methods=["GET"])
def get_moves():
    return jsonify(len(moves) + webgui_db[0][0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8282, debug=False)
