import numpy as np
import gym
from gym import  spaces
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D,Permute
from tensorflow.keras.optimizers import Adam
import tensorflow
print(tensorflow.version)
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy,LinearAnnealedPolicy
from rl.memory import SequentialMemory

#Board helper function
NUMBER_ROWS = 8
NUMBER_COLUMNS = 8
WINDOW_LENGTH = 4
#A starting board is 0's surrounded by the number 2. 1's filled in for snake locations
def initBoard(Nrows,NCols):
    snakeBoard = np.zeros([NUMBER_ROWS, NUMBER_COLUMNS])
    snakeBoard[:,0] = 3
    snakeBoard[0,:] = 3
    snakeBoard[NUMBER_ROWS-1,:] = 3
    snakeBoard[:,NUMBER_COLUMNS-1] = 3
    return snakeBoard

def denseNN():
    model = Sequential()
    model.add(Flatten(input_shape=(WINDOW_LENGTH,) + (NUMBER_ROWS, NUMBER_COLUMNS)))
    model.add(Dense(20))
    model.add(Activation('relu'))
    model.add(Dense(20))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    return model

def convNN():
    model = Sequential(name="snakeScreen")
    model.add(Conv2D(20, (3, 3), padding='same', strides=(1,1),activation='relu',input_shape=(WINDOW_LENGTH,NUMBER_ROWS,NUMBER_COLUMNS)))
    model.add(Conv2D(20, (3, 3), padding='same', strides=(1,1),activation='relu',input_shape=(WINDOW_LENGTH,NUMBER_ROWS,NUMBER_COLUMNS)))
    model.add(Flatten())
    model.add(Dense(54, activation='relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())
    return model

model = convNN()

class SnakeEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}


    def __init__(self, grid,currentLocation=np.array([0,0])):
        super(SnakeEnvironment, self).__init__()
        self.grid = grid
        self.currentLocation = currentLocation

        self.grid[tuple(self.currentLocation)] = 2
        self.MAX_GRID = self.grid.copy()
        self.MAX_GRID[self.MAX_GRID == 0] = 2
        self.current_step = 0
        self.gridSize = grid.shape

        self.reward_range = (0, self.gridSize[0]*self.gridSize[1])
        self.spacesToDirectionLabel = {0:'L',1:'U',2:'R',3:'D'}
        self.movementAmount = {'L':np.array([0,-1]),'R':np.array([0,1]),'U':np.array([-1,0]),'D':np.array([1,0])}
        self.costEatingSelf = 5 #Reduce reward when moving onto self.
        self.currentLength = 1
        # Move Left:1,R:2,U:3,D:4.
        self.action_space = spaces.Discrete(4)

        # Snake Grid
        self.observation_space = spaces.Box(low=self.grid,high=self.MAX_GRID,dtype=np.int16)

        #Results
        self.finalLengths = []
        self.finalRewards = []

    def _next_observation(self):
        return self.grid

    def _moveSnake(self, action):
        # Action where the snake head moves 1 space over.
        new_location = self.currentLocation + self.movementAmount[self.spacesToDirectionLabel[action]]
        #Do nothing actions (On edge and moving towards edge - successful move, with no reward
        if self.grid[tuple(new_location)] == 3:
            return 0

        if self.grid[tuple(new_location)] == 1:
            print("Failed to complete. Ate self.")
            return -1
        else:
            self.grid[tuple(self.currentLocation)] = 1
            self.grid[tuple(new_location)] = 2
            self.currentLocation = new_location
            self.currentLength += 1
            return 1

    def step(self, action):
        print("Action:",action)
        # Execute one time step within the environment
        current_reward = self._moveSnake(action) #returns 0=No move, 1=moved successfully, -1=Moved into itself

        self.current_step += 1

        reward = self.currentLength/self.current_step

        done = False
        numZeros = len(self.grid[self.grid == 0])

        if numZeros == 0:
            done = True
        if current_reward < 0:
            reward = -1
            done = True
        if self.current_step == 30:
            done = True

        if done:
            print("Final length:", self.currentLength)
            self.finalLengths.append(self.currentLength)
            self.finalRewards.append(self.currentLength/self.current_step)
        obs = self._next_observation().copy()
        return obs, reward, done, {}


    def reset(self):
        print("State:", self.grid)
        self.currentLength = 1
        self.current_step=1
        #Random start location. Allow for non square grids by getting separate ranges
        self.currentLocation = np.array([np.random.randint(1,self.gridSize[0]-1),
                                         np.random.randint(1,self.gridSize[1]-1)
                                        ])

        self.grid = initBoard(NUMBER_ROWS, NUMBER_COLUMNS)
        self.grid[tuple(self.currentLocation)] = 1
        self.reward = 1
        return self.grid

    def render(self, mode='human', close=False):
        # print("Render grid:",self.grid)
        return self.grid

snakeBoard = initBoard(NUMBER_ROWS, NUMBER_COLUMNS)
snakeEnv = SnakeEnvironment(snakeBoard,np.array([1,1]))

np.random.seed(123)
snakeEnv.seed(123)
nb_actions = snakeEnv.action_space.n


# model = denseNN()
model = convNN()
policy = EpsGreedyQPolicy()
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                              attr='eps',
                              value_max=1.,
                              value_min=.1,
                              value_test=.05,
                              nb_steps=20000)
memory = SequentialMemory(limit=200, window_length=WINDOW_LENGTH)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=20,target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mse'])
dqn.fit(snakeEnv,nb_steps=20000, visualize=True, verbose=2)

import matplotlib.pyplot as plt

plt.figure()
plt.subplot(121)
plt.plot(snakeEnv.finalRewards)
plt.subplot(122)
plt.plot(snakeEnv.finalLengths)
plt.show()
# dqn.test(snakeEnv, nb_episodes=1, visualize=True)



