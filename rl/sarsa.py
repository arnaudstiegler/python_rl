import numpy as np

def discretize(value,factor,space_size):
    return(int((float(factor)*value)/space_size))

class sarsa_Agent:

    def __init__(self,width,height,num_creep):

        self.discretization_factor = 25
        self.alpha = 0.3
        self.gamma = 0.99

        width_discretized = discretize(width,self.discretization_factor,width)
        height_discretized = discretize(height,self.discretization_factor,height)

        self.Q = np.random.rand((width_discretized+1,height_discretized+1,num_creep)) #+1 pour le cas où y'a pas de creep

    def take_action(self,obs):

