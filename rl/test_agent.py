from ple.games.pixelcopter import Pixelcopter
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
    if(len(list) == 7):
        out = np.array(list)
        return out


game = Pixelcopter(500,500)
p = PLE(game, fps=30, display_screen=True, force_fps=False)
p.init()

scores = []


agent = DQL.deep_learner(1,2,p.getActionSet())
agent.learning_rate = 0.001
agent.epsilon = 0.001
agent.epsilon_decay = 1.0
agent.epsilon_min = 0.00001
#agent.load_model_json()
agent.load_alt()

nb_games = 0


for layer in agent.model.layers:
    print(layer.get_config())

while(nb_games < 20):
    if p.game_over(): #check if the game is over
        scores.append(game.getScore())
        nb_games +=1
        p.reset_game()

    if(nb_games % (50) == 0):
        print(nb_games)

    obs = process_obs(game.getGameState())
    action = agent.act(obs)
    reward = p.act(action)


plt.plot(range(len(scores)),scores)
plt.show()
