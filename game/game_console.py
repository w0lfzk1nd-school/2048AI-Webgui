import sys
import json
import questionary
from game.game_engine import Game2048

def print_board(board):
    for row in board:
        print("+----" * len(row) + "+")
        print("|".join(f"{num:4d}" if num > 0 else "    " for num in row) + "|")
    print("+----" * len(row) + "+")

def save_game_data(game_data, filename="datasets/hooman.json"):
    with open(filename, 'w') as f:
        json.dump(game_data, f)
    print(f"Game data saved to {filename}")

def main():
    game = Game2048()
    game_data = []

    print_board(game.board)

    while not game.is_game_over():
        move = input("Enter move (w, a, s, d): ").strip().lower()
        if move in ['w', 'a', 's', 'd']:
            game_data.append((game.board, move))
            game.move(move)
            print_board(game.board)
        elif move == 'exit':
            save_game_data(game_data)
            sys.exit("Game has been exited.")
        elif move == 'restart':
            game = Game2048()
            game_data = []
            print_board(game.board)
        else:
            print("Invalid move. Please only use w, a, s, or d.")

    save_game_data(game_data)
    print("Game over! Final board:")
    print_board(game.board)

if __name__ == "__main__":
    main()
