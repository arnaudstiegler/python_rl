from ple.games import Catcher
from ple import PLE
import random
import matplotlib.pyplot as plt
import numpy as np
import rl.deep_Q_learning as DQL
import h5py



def process_obs(obs):
    list = []
    max_values = np.array([])
    for item in obs:
        list.append(obs[item])
    if(len(list) == 4):
        out = np.array([list])
        return out


game = Catcher(500,500)
p = PLE(game, fps=30, display_screen=False, force_fps=False)
p.init()
scores = []
length = []
print(p.getActionSet())

agent = DQL.deep_learner(4,3,p.getActionSet())

nb_games = 0
nb_max = 500

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

        if (nb_games % (nb_max / 10) == 0):
            print('iteration:' +str(nb_games))
            print(reward)
        reward = 0




    obs = process_obs(game.getGameState())
    action = agent.act(obs)
    reward += p.act(action)
    step += 1
    agent.remember(obs,action,reward,process_obs(game.getGameState()),p.game_over())


agent.model.save_weights('my_model.h5')
plt.figure()
plt.plot(range(len(scores)),scores)
plt.show()
plt.figure()
plt.plot(range(len(length)),length)
plt.show()