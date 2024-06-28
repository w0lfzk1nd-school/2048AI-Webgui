import os
import sys
import numpy as np
from utils.utilities import print_and_log, form_time, log, gen_key
from game.game_engine import Game2048
import matplotlib.pyplot as plt
import questionary
from time import sleep


class Environment:
    global calc_reward

    def __init__(self):
        self.game = Game2048()
        self.model = ""
        self.runtime = 0
        self.total_games = 0
        self.total_scores = []
        self.total_rewards = []
        self.last_actions = []
        self.last_action = None
        self.last_reward = 0
        self.game_steps = 0
        self.total_steps = 0
        self.unsuccessful_moves_count = 0

        self.runtime_history = []
        self.action_history = []
        self.reward_history = []
        self.score_history = []
        self.step_history = []
        self.highscore_history = []
        self.bestblock_history = []
        self.average_reward_history = []
        self.average_score_history = []
        self.average_steps_history = []
        self.average_time_history = []

    def reset(self):
        self.game = Game2048()
        return np.array(self.game.board)

    def step(self, action):
        max_threshhold = -512
        max_steps = 3000
        directions = ["w", "a", "s", "d"]
        direction = directions[action]
        old_score = self.game.score
        
        self.game.move(direction)
        
        new_board = np.array(self.game.board)
        new_score = self.game.score
        max_value = np.max(self.game.board)

        if old_score < new_score:
            reward = new_score - old_score
        else:
            reward = -1
        
        self.total_steps += 1
        self.game_steps += 1
        done = self.game.is_game_over()
        
        if self.game_steps == max_steps:
            done = True
        
        self.last_action = direction
        self.last_reward = round(reward, 3)
        self.total_rewards.append(reward)
        
        if done:
            if max_value >= 2048:
                reward += 1000
            else:
                reward = max_threshhold

            self.total_games += 1
            self.step_history.append(self.game_steps)
            self.game_steps = 0
            self.runtime_history.append(self.runtime)
            self.total_scores.append(new_score)
            self.score_history.append(self.game.score)
            self.action_history.append(direction)
            self.reward_history.append(reward)
            self.highscore_history.append(self.game.tot_highscore)
            self.bestblock_history.append(self.game.bestblock)
            self.average_reward_history.append(
                sum(self.total_rewards) / len(self.total_rewards)
            )
            self.average_time_history.append(
                sum(self.runtime_history) / len(self.runtime_history)
            )
            self.average_score_history.append(
                sum(self.total_scores) / len(self.total_scores)
            )
            self.average_steps_history.append(
                sum(self.step_history) / len(self.step_history)
            )
            print_and_log(
                f"[DONE]>> Game {self.total_games} finished. Final score: {new_score}"
            )

        reward = float(reward)
        
        return new_board, reward, done

    def render(self, scores, traintime):
        highscore = scores["high"]
        bestscore = scores["best"]
        console_output = f"\033[H\033[J"  # Clear screen and move cursor to the top
        console_output += f"\nTraining-Time: {form_time(traintime)}\nMove: {self.game_steps} | Score: {self.game.score} | Highscore: {highscore} | BestBlock: {bestscore}\nBoard:\n"
        for row in self.game.board:
            console_output += f"{' | '.join(str(num).ljust(4) for num in row)}\n"
            console_output += f"{'-' * 25}\n"

        # Ergänzende Informationen, die auch ins Log sollen
        extended_info = (
            f"Last Action: {self.last_action}, Last Reward: {self.last_reward}"
        )
        if self.total_games == 0:
            tot_games = 1
        else:
            tot_games = self.total_games
        extended_info += f", AvgReward: {sum(self.total_rewards) / tot_games:.2f}, Avg.Points: {sum(self.total_scores) / tot_games:.2f}\n"
        extended_info += f"Total Games: {self.total_games}, Total Steps: {self.total_steps}\n==================================================\n"

        sys.stdout.write(console_output + extended_info)
        sys.stdout.flush()

#    def render(self, scores, traintime):
#        highscore = scores["high"]
#        bestscore = scores["best"]
#        console_output = f"\nTraining-Time: {form_time(traintime)}\nMove: {self.game_steps} | Score: {self.game.score} | Highscore: {highscore} | BestBlock: {bestscore}\nBoard:\n"
#        for row in self.game.board:
#            console_output += f"{' | '.join(str(num).ljust(4) for num in row)}\n"
#            console_output += f"{'-' * 25}\n"
#
#        # Ergänzende Informationen, die auch ins Log sollen
#        extended_info = (
#            f"Last Action: {self.last_action}, Last Reward: {self.last_reward}"
#        )
#        if self.total_games == 0:
#            tot_games = 1
#        else:
#            tot_games = self.total_games
#        extended_info += f", AvgReward: {sum(self.total_rewards) / tot_games:.2f}, Avg.Points: {sum(self.total_scores) / tot_games:.2f}\n"
#        extended_info += f"Total Games: {self.total_games}, Total Steps: {self.total_steps}\n==================================================\n"
#
#        print_and_log(console_output + extended_info)

    def plot_history(self):
        plt.figure(figsize=(10, 12))

        form_runtime = [round((runtime_entry / 1000), 2) for runtime_entry in self.runtime_history]

        plt.subplot(411)
        plt.plot(form_runtime, label="Time per Game")
        plt.plot(self.average_time_history)
        plt.xlabel("Games")
        plt.ylabel("Time (s)")
        plt.legend()

        plt.subplot(412)
        plt.plot(self.step_history, label="Steps per Game")
        plt.plot(self.average_steps_history, label="Average Steps")
        plt.xlabel("Games")
        plt.ylabel("Steps")
        plt.legend()

        plt.subplot(413)
        plt.plot(self.score_history, label="Score per Game")
        plt.plot(self.average_score_history, label="Average Score")
        plt.xlabel("Games")
        plt.ylabel("Score")
        plt.legend()

        plt.subplot(414)
        plt.plot(self.reward_history, label="Reward per Game")
        plt.plot(self.average_reward_history, label="Average Reward")
        plt.xlabel("Games")
        plt.ylabel("Reward")
        plt.legend()

        plt.tight_layout()

        if questionary.confirm("Möchten Sie diesen Plot als Bild speichern?").ask():
            save_path = "saves"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_path = os.path.join(save_path, f"{self.model.replace('-data.json', '')}_plot1.png")
            plt.savefig(file_path, bbox_inches="tight")
            print(f"Plot 1 wurde gespeichert unter: {file_path}")

        plt.figure(figsize=(10, 12))  # Ein neues Figure-Objekt für die nächsten Plots

        plt.subplot(311)
        plt.plot(self.action_history, label="Last Action")
        plt.xlabel("Games")
        plt.ylabel("Action")
        plt.legend()
        
        plt.subplot(312)
        plt.plot(self.highscore_history, label="Highscores")
        plt.xlabel("Games")
        plt.ylabel("Highscores")
        plt.legend()
        
        plt.subplot(313)
        plt.plot(self.bestblock_history, label="Highest Block Value")
        plt.xlabel("Games")
        plt.ylabel("Block Value")
        plt.legend()

        plt.tight_layout()

        if questionary.confirm("Möchten Sie diesen Plot als Bild speichern?").ask():
            save_path = "saves"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_path = os.path.join(save_path, f"{self.model.replace('-data.json', '')}_plot2.png")
            plt.savefig(file_path, bbox_inches="tight")
            print(f"Plot 2 wurde gespeichert unter: {file_path}")

        plt.show()