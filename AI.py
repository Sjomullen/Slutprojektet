import gym
import numpy as np

env = gym.make('BreakoutDeterministic-v4')

frame = env.reset()

env.render()



is_done = False

while not is_done:

    frame, reward, is_done, _ = env.step(env.action_space.sample())

    env.render()

def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)   


def downsample(img):
    return img[::2, ::2]

def preprocess(img):
    return to_grayscale(downsample(img))

def transform_reward(reward):
    return np.sign(reward)

def fit_batch(model, gamma, start_states, actions, rewards, next_states, is_terminal):
                """Do one deep Q learning iteration.


Params:
- model: The DQN
- gamma: Discount factor (should be 0.99)
- start_states: numpy array of starting states 
- actions: numpy array of one-hot encoded actions corresponding to the start states
- rewards: numpy array of rewards corresponding to the start and actions
- next_states: numpy array of the resulting states corresponding to the start states and actions
- is_terminal: numpy boolean array of wether the resulting state is terminal


"""


# First, predict the Q values of the next states. Note we are passing one as the mask.
next_Q_values = model.predict([next_states, np.ones(action.shape)])
# The Q values of the terminal states is 0 by definition, so override them
next_Q_values[is_terminal] = 0
# The Q values of each start state is the reawrd + gamma * the max next state of Q value
Q_values = rewards + gamma * np.max(next_Q_values, axis=1)
# Fit the keras model. Note how we are passing the actions as the mask and mulitplying 
# the targets by the actions.
model.fit(
    [start_states, actions], actions * Q_values[:, None],
    nb_epoch=1, batch_size=len(start_states), verbose=0

)