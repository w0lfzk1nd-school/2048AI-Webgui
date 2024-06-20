import ollama
import random

class OllamaSolve:
    def __init__(self, model_name="llama3"):
        self.model_name = model_name
        self.initialprompt = """
### 2048 Game Choice Maker

#### Goal:
Your goal is to analyze the best move for the given boardlayout and tileposition, with the best outcome for this move and the following moves.
Good moves are when you combine two tiles with the same value, move tiles in the last row or keep the bigger numbers in a corner of the last row.

#### Format of Input:
You will receive the current board layout as a 4x4 matrix represented in Python list format. Each cell can contain either 0 (empty) or a power of 2 tile.

#### Format of Output:
Your response should be a single number indicating the direction of the move: 0 for Up, 1 for Left, 2 for Down, and 3 for Right. You are strictly forbidden to generate any additional text beside the chosen number.

### Structure of Conversation and Regulations:
The conversation will follow a simple back-and-forth pattern.
You, as the model, will receive the current board layout from the user as input.
After processing the input, you will reply with the number corresponding to the best possible move for that board.
Your task is to consistently analyze each board layout and determine the optimal move with the most combinations of tiles and moving the biggest numbers in the last row (down).
You are not allowed to generate code, other information, notes, or any other text.
You are not allowed to end or comment the game at any time.

#### Example Conversation:
user = the human.
assistant = you.
```
user: "[[0, 0, 0, 4], [0, 0, 0, 2], [0, 0, 0, 0], [0, 0, 4, 2]]"
assistant: 2
user: "[[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4], [0, 0, 4, 4]]"
assistant: 2
user: "[[0, 0, 0, 0], [0, 4, 0, 0], [0, 0, 0, 0], [0, 0, 8, 8]]"
```

From now on, after your next message, the conversation for the game will start. Please confirm and shorty summarize the task and continue the conversation after my next message.
"""
        self.intialconfirm = """Got it! Let's summarize: you'll provide the current board layout as a 4x4 matrix in Python list format, consisting of zeros (empty cells) and power of 2 tiles. My task is to determine the optimal move based on the provided layout, focusing on combining tiles and following the win strategy. I'll reply with a single number indicating the direction of the recommended move: 0 for Up, 1 for Left, 2 for Down, and 3 for Right. Let's start the game with your next board layout!"""
        self.messages = [
            {"role": "user", "content": self.initialprompt},
            {"role": "assistant", "content": self.intialconfirm},
            {"role": "user", "content": "[[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 4, 8]]"},
            {"role": "assistant", "content": "2"}
        ]

    def add_user_message(self, content):
        self.messages.append({"role": "user", "content": content})

    def get_bot_response(self):
        try:
            response = ollama.chat(model=self.model_name, messages=self.messages)
        except Exception as e:
            print(f">> OLLAMA intern ERROR: {e}")
            response = random.randrange(4)
        bot_reply = response["message"]["content"]
        self.messages.append({"role": "assistant", "content": bot_reply})
        return bot_reply

    def start_chat(self):
        print("Start des Chats (gib 'exit' ein, um zu beenden)")
        while True:
            user_input = input("Du: ")
            if user_input.lower() == "exit":
                break
            self.add_user_message(user_input)
            bot_reply = self.get_bot_response()
            print(f"Bot: {bot_reply}")
            
    def solve_board(self, user_input):
        self.add_user_message(user_input)
        bot_reply = self.get_bot_response()
        return bot_reply

    def reset_chat(self):
        self.messages = [
            {"role": "user", "content": self.initialprompt},
            {"role": "assistant", "content": self.intialconfirm},
            {"role": "user", "content": "[[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 4, 8]]"},
            {"role": "assistant", "content": "2"}
        ]
        print("Konversation zurückgesetzt.")


if __name__ == "__main__":
    chat = OllamaSolve()
    chat.start_chat()
