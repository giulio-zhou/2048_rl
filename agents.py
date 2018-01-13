import gym
import gym_2048
import numpy as np
import time

def run_random_agent(game):
    action_space = ['up', 'down', 'left', 'right']
    i = 0
    while True:
        action = np.random.choice(action_space)
        board, reward, close, _ = game.step(action)
        print(game)
        i += 1

game = gym.make('2048-v0')
run_random_agent(game)
