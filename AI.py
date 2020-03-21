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

def atari_model(n_actions):
    # We assume a theano backend here, so the "channels" are first.
    ATARI_SHAPE = (4, 105, 80)

    # With the functional API we need to define the inputs
    frames_input = keras.layers.Input(ATARI_SHAPE, name='frames')
    actions_input = keras.layers.Input((n_actions,), name='mask')

    # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0,1].
    normalized = keras.layers.Lambda(lambda x: x / 255.0)(frames_input)

    # "The first hidden layer convolves 16 8x8 filters with stride 2, again followed by a rectifer nonlinearity"
    conv_1 = keras.layers.convolutional.Convolution2D(16, 8, 8, subsample=(4, 4), activation='relu')(conv_1)
    
    # "The second hidden layer convolves 32 4x4 filters with stride 2, again followed by a rectifier nonlinearity."
    conv_2 = keras.layers.convolutional.Convolution2D(32, 4, 4, subsample=(2, 2), activation='relu')(conv_1)

    # Flatteningthe second convolutional layer.
    conv_flattened = keras.layers.core.Flatten()(conv_2)

    # "The final hidden layer is fully-connected and consists of 256 rectifier units."
    hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)

    # "The output layer is a fullt-connected linear layer with a single output for each valid action."
    output = keras.layers.Dense(n_actions)(hidden)

    # "Finally, we multiply the output by the mask!"
    filtered_output = keras.layers.merge([output, actions_input], mode='mul')


    self.model = keras.models.Model(input=[frames_input, actions_input], output=filtered_output)
    optimizer = keras.optimizer.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    self.model.compile(optimizer, loss='mse')


class RingBuf:
    def _init_(self, size):
        




