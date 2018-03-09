from ple.games.pixelcopter import Pixelcopter
from ple import PLE
import random
import matplotlib.pyplot as plt
import numpy as np
import rl.deep_Q_learning as DQL
import h5py




game = Pixelcopter(500,500)
p = PLE(game, fps=30, display_screen=True, force_fps=False)
p.init()

def process_obs(obs):
    list = []
    max_values = np.array([])
    for item in obs:
        list.append(obs[item])
    if(len(list) == 7):
        out = np.array([list])/np.array([500,500,500,500,500,500,500])
        return out

scores = []


agent = DQL.deep_learner(7,2,p.getActionSet())

nb_games = 0
nb_max = 1000

for layer in agent.model.layers:
    print(layer.get_config())

step = 0
distance = []
reward = 0

while(nb_games < nb_max):


    if p.game_over(): #check if the game is over
        if(len(agent.memory) > 1):
            agent.replay(int(len(agent.memory)*0.8))
        scores.append(reward)
        nb_games +=1
        distance.append(step)
        step = 0
        p.reset_game()

    if(nb_games % (10) == 0):
        print(nb_games)

    obs = process_obs(game.getGameState())
    action = agent.act(obs)
    reward = p.act(action) + step
    step +=1
    if p.game_over() is False:
        agent.remember(obs,action,reward,process_obs(game.getGameState()),p.game_over())


agent.save_alt()
plt.figure()
plt.plot(range(len(scores)),scores)
plt.show()

plt.figure()
plt.plot(range(len(distance)),distance)
plt.show()




