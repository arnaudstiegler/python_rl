import numpy as np
import collections
from keras import Sequential
from keras import layers,optimizers
from keras.models import model_from_json
import h5py



# Deep Q-learning Agent
class deep_learner:
    def __init__(self, state_size, action_size,action_table):
        self.state_size = state_size
        self.action_size = action_size
        self.action_table = action_table
        self.memory = collections.deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.1 # exploration rate
        self.epsilon_min = 0.0001
        self.epsilon_decay = 0.99
        self.learning_rate = 0.1
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(layers.Dense(12, input_dim=self.state_size,activation='relu'))
        model.add(layers.Dense(12, activation='relu'))
        #model.add(layers.Dense(8, activation='relu'))
        model.add(layers.Dense(3, activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam(lr = self.learning_rate,beta_1=0.9,beta_2=0.999))

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            out = np.random.randint(0,3)
            if out == 1:
                return 97
            elif out == 2:
                return 100
            else:
                return 0
        act_values = self.model.predict(state)
        if(np.argmax(act_values[0]) == 0):
            return 119
        else:
            return 0

    def act_random(self):
        rand = np.random.randint(0, 3)
        if rand == 1:
            return 97
        elif rand == 2:
            return 100
        else:
            return 0

    def replay(self, batch_size):
        minibatch = []
        index_minibatch = np.random.randint(0,len(self.memory)-1,batch_size)
        for index in index_minibatch.tolist():
            minibatch.append(self.memory[index])
        minibatch = np.array(minibatch)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            if(action == 97):
                #target_f[0][action]
                target_f[0][0] = target #because index 119 -> problem
            elif(action == 100):
                target_f[0][1] = target
            else:
                target_f[0][2] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model_json(self):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model.h5")
        print("Saved model to disk")

    def save_alt(self):
        json_string = self.model.to_json()
        open('test.json', 'w').write(json_string)
        self.model.save_weights('weight_test.h5')

    def load_alt(self):
        model = model_from_json(open('test.json').read())
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        model.load_weights('weight_test.h5')
        self.model = model

    def load_model_json(self):
        # load json and create model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        self.model = loaded_model
        self.model.compile(loss='mse', optimizer=optimizers.Adam(lr=self.learning_rate))

        # load weights into new model
        self.model.load_weights("model.h5")
        print("Loaded model from disk")

