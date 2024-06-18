import random

class Game2048:
    def __init__(self, size=4):
        self.size = size
        self.board = [[0] * size for _ in range(size)]
        self.score = 0
        self.highscore = 0
        self.tot_highscore = 0
        self.bestblock = 0
        self.add_new_tile()
        self.add_new_tile()

    def add_new_tile(self):
        empty_cells = [(x, y) for x in range(self.size) for y in range(self.size) if self.board[x][y] == 0]
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x][y] = random.choice([2, 4])

    def move(self, direction):
        old_board = [row[:] for row in self.board]
        def merge(row):
            non_zero = [i for i in row if i != 0]
            merged = []
            i = 0
            while i < len(non_zero):
                if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                    merged_score = non_zero[i] * 2
                    self.score += merged_score
                    
                    if self.score > self.highscore:
                        self.highscore = self.score
                        
                    if merged_score > self.bestblock:
                        self.bestblock = merged_score

                    merged.append(non_zero[i] * 2)
                    i += 2
                else:
                    merged.append(non_zero[i])
                    i += 1
            return merged + [0] * (self.size - len(merged))

        def transpose(board):
            return [list(row) for row in zip(*board)]

        def reverse(board):
            return [row[::-1] for row in board]

        if direction == 'w' or direction == 0:
            self.board = transpose(self.board)
            self.board = [merge(row) for row in self.board]
            self.board = transpose(self.board)
        elif direction == 's' or direction == 2:
            self.board = transpose(self.board)
            self.board = reverse(self.board)
            self.board = [merge(row) for row in self.board]
            self.board = reverse(self.board)
            self.board = transpose(self.board)
        elif direction == 'a' or direction == 1:
            self.board = [merge(row) for row in self.board]
        elif direction == 'd' or direction == 3:
            self.board = reverse(self.board)
            self.board = [merge(row) for row in self.board]
            self.board = reverse(self.board)
            
        if self.board != old_board:  # Überprüfen, ob sich das Board geändert hat
            self.add_new_tile()



    def is_game_over(self):
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x][y] == 0:
                    return False
                if x < self.size - 1 and self.board[x][y] == self.board[x + 1][y]:
                    return False
                if y < self.size - 1 and self.board[x][y] == self.board[x][y + 1]:
                    return False
        return True

