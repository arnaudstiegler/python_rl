from ple.games import FlappyBird
from ple import PLE
import random
import matplotlib.pyplot as plt
import numpy as np
import rl.deep_Q_learning as DQL
import h5py



def process_obs(obs):
    list = []
    for item in obs:
        list.append(obs[item])
    y_position = list[0]
    pipe_1_top_y = list[3]
    pipe_1_bottom_y = list[4]
    list[3] = pipe_1_top_y - y_position
    list[4] = pipe_1_bottom_y - y_position

    pipe_2_top_y = list[6]
    pipe_2_bottom_y = list[7]
    list[6] = pipe_2_top_y - y_position
    list[7] = pipe_2_bottom_y - y_position

    if(len(list) == 8):
        out = np.array([list])
        return out


game = FlappyBird()
p = PLE(game, fps=30, display_screen=False, force_fps=False)
p.init()
scores = []
length = []


agent = DQL.deep_learner(8,2,p.getActionSet())

nb_games = 0
nb_max = 1000

for layer in agent.model.layers:
    print(layer.get_config())

step = 0
reward = 0

while(nb_games < nb_max):
    if p.game_over(): #check if the game is over
        if(len(agent.memory) < 150):
            agent.replay(int(len(agent.memory)*0.8))
        else:
            agent.replay(150)
        scores.append(reward)
        length.append(step)
        step = 0
        nb_games +=1
        p.reset_game()

    if(nb_games % (nb_max/10) == 0):
        print(nb_games)

    obs = process_obs(game.getGameState())
    action = agent.act(obs)
    reward = p.act(action) + step
    step += 1
    agent.remember(obs,action,reward,process_obs(game.getGameState()),p.game_over())


agent.model.save_weights('my_model.h5')
plt.figure()
plt.plot(range(len(scores)),scores)
plt.show()
plt.figure()
plt.plot(range(len(length)),length)
plt.show()