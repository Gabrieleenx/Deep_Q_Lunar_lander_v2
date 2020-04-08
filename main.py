import random
from _collections import deque
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
env = gym.make("LunarLander-v2")
'''
env.reset()
action_space = env.action_space
observation = env.observation_space
print(action_space)
print(observation)
for i in range(1000):
    env.render()
    env.step(random.choice([0,1,2,3]))
env.close()
'''

class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experiance):
        self.buffer.append(experiance)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)
        return [self.buffer[i] for i in index]


memory = Memory(max_size=1000000)

possible_actions = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

input_size = 8
output_size = 4
learning_rate = 0.0005
pretrain_length = 100
batch_size = 64
num_games = 500
training = True
decay_rate = 0.0001
explore_high = 0.02
explore_low = 0.01
gamma = 0.99
'''
model = tf.keras.Sequential([
    layers.Dense(200, input_dim=input_size, activation=keras.activations.relu, kernel_initializer=tf.keras.initializers.glorot_normal),
    layers.Dense(200, activation=keras.activations.relu, kernel_initializer=tf.keras.initializers.glorot_normal),
    layers.Dense(output_size, activation=None, kernel_initializer=tf.keras.initializers.glorot_normal)
])
'''
model = tf.keras.Sequential([
    layers.Dense(200, input_dim=input_size, activation=keras.activations.relu),
    layers.Dense(200, activation=keras.activations.relu),
    layers.Dense(output_size, activation=None)
])


def loss(target_Q, actions_value):
    loss = tf.reduce_mean(tf.square(target_Q - actions_value), axis=1)
    return loss


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='mse',
              metrics=['accuracy'])
del model

model = tf.keras.models.load_model("save_")


if training == True:
    observation = env.reset()
    for i in range(pretrain_length):
        action = random.choice([0, 1, 2, 3])
        next_observation, reward, done, _ = env.step(action)
        env.render()
        if done:

            next_observation = np.zeros(len(observation))
            memory.add((observation, action, reward, next_observation, done))

            observation = env.reset()

        else:
            memory.add((observation, action, reward, next_observation, done))
            observation = next_observation

    step =0
    for i in range(num_games):
        score = 0
        observation = env.reset()
        done = False
        while not done:
            Q_val = 0
            step += 1
            random_action = np.random.rand()
            explore_action = explore_low + (explore_high-explore_low)*np.exp(-decay_rate * step)
            if random_action < explore_action:
                action = random.choice([0, 1, 2, 3])
            else:
                obs = np.array([observation])
                Q_val = model.predict(obs)
                action = np.argmax(Q_val)
            next_observation, reward, done, _ = env.step(action)
            if i % 10 == 0:
                env.render()
            score += reward
            if done:
                next_observation = np.zeros(len(observation))
                memory.add((observation, action, reward, next_observation, done))
            else:
                memory.add((np.float32(observation), np.int(action), np.float32(reward), np.float32(next_observation), done))
                observation = next_observation
            pass

            # trainging
            batch = memory.sample(batch_size)
            observations_batch = np.array([each[0] for each in batch])
            action_batch = np.array([each[1] for each in batch])
            reward_batch = np.array([each[2] for each in batch])
            next_observation_batch = np.array([each[3] for each in batch])
            done_batch = np.array([each[4] for each in batch])

            target = model.predict(observations_batch)
            next_prediction = model.predict(next_observation_batch)
            target_Q = target.copy()  # not fully sure if nessesery
            batch_index = np.arange(batch_size)

            target_Q[batch_index, action_batch] = reward_batch + gamma*np.max(next_prediction, axis=1)
            model.fit(observations_batch, target_Q, verbose=0)
        if i % 10 == 0:
            model.save("save_")
            print('model saved')
        print('episode', i, score, explore_action)
