from abc import ABC
from typing import List, Tuple
from random import sample, uniform, randint

import time

import gym

import tensorflow as tf
from tensorflow import Tensor, convert_to_tensor, GradientTape, one_hot, float64, Module, cast, function
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import Loss, Huber
from tensorflow.keras.optimizers import Optimizer, Adam
from tensorflow.keras.regularizers import l2

from numpy import ndarray, newaxis, argmax, append
from statistics import mean


class DDDQN(Model, ABC, Module):
    _d1: Dense
    _d2: Dense
    _val: Dense
    _adv: Dense

    def __init__(self):
        super(DDDQN, self).__init__()
        self._d1 = Dense(64, activation='relu', kernel_regularizer=l2(REGULARIZATION_FACTOR))
        self._d2 = Dense(64, activation='relu', kernel_regularizer=l2(REGULARIZATION_FACTOR))
        self._val = Dense(1, activation='linear')
        self._adv = Dense(env.action_space.n, activation='linear')

    def call(self, x: Tensor, training: bool = False, mask=None) -> Tensor:
        x = self._d1(x)
        x = self._d2(x)
        val: Tensor = self._val(x)
        adv: Tensor = self._adv(x)
        Q: Tensor = val + adv - tf.math.reduce_mean(adv, axis=1, keepdims=True)
        return Q

    def advantage(self, x: Tensor or ndarray) -> Tensor:
        x = self._d1(x)
        x = self._d2(x)
        a: Tensor = self._adv(x)
        return a


class ReplayBuffer:
    _buffer_size: int
    _memory: List[Tuple[ndarray, int, float, ndarray, bool]]

    def __init__(self, buffer_size: int = 1000000):
        self._buffer_size = buffer_size
        self._memory = []

    @property
    def size(self) -> int:
        return len(self._memory)

    def add(self, state: ndarray, action: int, reward: float, observation: ndarray, done: bool):
        experience: Tuple[ndarray, int, float, ndarray, bool] = (state, action, reward, observation, done)
        self._memory.append(experience)
        if self.size > self._buffer_size:
            self._memory = sample(self._memory, int(self._buffer_size * 0.6))

    def sample_batch(self, batch_size: int = 64) -> List[Tuple[ndarray, int, float, ndarray, bool]]:
        batch: List[Tuple[ndarray, int, float, ndarray, bool]] = sample(self._memory, batch_size)
        return batch


class Agent:
    _replay_buffer: ReplayBuffer
    _q_model: DDDQN
    _target_model: DDDQN
    _model_version: int
    _epsilon: float
    _episode_rewards: List[float]

    def __init__(self, q_net: DDDQN):
        self._replay_buffer = ReplayBuffer()
        self._q_model = q_net
        self._target_model = DDDQN()
        self._target_model.set_weights(self._q_model.get_weights())
        self._model_version = 0
        self._epsilon = EPSILON
        self._episode_rewards = []

    def _update_epsilon(self):
        self._epsilon = max(self._epsilon * EPSILON_DECAY_RATE, MIN_EPSILON)

    def _update_episode_rewards(self, episode_reward: float):
        self._episode_rewards.append(episode_reward)
        while len(self._episode_rewards) > 50:
            self._episode_rewards.pop(0)

    def _average_reward(self) -> float:
        return mean(self._episode_rewards)

    def _policy(self, x: Tensor or ndarray) -> int or ndarray[int]:
        if self._epsilon < uniform(0, 1):
            action: ndarray[int] = argmax(self._q_model.advantage(x))
            return action
        else:
            action: int = randint(0, 3)
            return action

    def _update_target_model(self):
        self._target_model.set_weights(self._q_model.get_weights())

    def _compute_q_targets(self, states: Tensor, actions: Tuple[int], rewards: Tuple[float], observations: Tensor,
                           dones: Tuple[bool]) -> Tensor:
        q: Tensor = self._q_model(states)
        next_q: Tensor = self._q_model(observations)
        next_q_tm: Tensor = self._target_model(observations)
        max_actions: ndarray[int] = argmax(next_q, axis=1)
        targets: List = []
        for index, action in enumerate(max_actions):
            if dones[index]:
                target_val: float = rewards[index]
            else:
                target_val: float = rewards[index] + GAMMA * next_q_tm[index, action] - q[index, actions[index]]
            q_target: Tensor = cast(q[index], dtype=float64) + one_hot(
                actions[index], env.action_space.n, on_value=cast(target_val, dtype=float64))
            targets.append(q_target)
        targets: Tensor = convert_to_tensor(targets, dtype=float64)
        return targets

    @function
    def _train_step(self, states: Tensor, q_targets: Tensor):
        with GradientTape() as tape:
            Q: Tensor = self._q_model(states)
            loss: Tensor = huber(q_targets, Q)
        grads: List[Tensor] = tape.gradient(loss, self._q_model.trainable_variables)
        adam.apply_gradients(zip(grads, self._q_model.trainable_variables))

    @staticmethod
    def _preprocess(batch: List[Tuple[ndarray, int, float, ndarray, bool]]
                    ) -> Tuple[Tensor, Tuple[int], Tuple[float], Tensor, Tuple[bool]]:
        data: List = []
        for iter_obj in zip(*batch):
            data.append(iter_obj)
        states: Tensor = convert_to_tensor(data[0], dtype=float64)
        actions: Tuple[int] = data[1]
        rewards: Tuple[float] = data[2]
        observations: Tensor = convert_to_tensor(data[3], dtype=float64)
        dones: Tuple[bool] = data[4]
        return states, actions, rewards, observations, dones

    def _build(self):
        test_input: ndarray = env.reset()
        test_input = append(test_input, 0.)
        test_input = test_input[newaxis, ...]
        self._q_model(test_input)
        self._target_model(test_input)

    def training(self):
        self._build()
        start: float = time.time()
        step_count: int = 0
        for episode in range(MAX_EPISODES):
            episode_reward: float = 0.
            state: ndarray = env.reset()
            state = append(state, 0.)
            state: ndarray = state[newaxis, ...]
            for step in range(1, MAX_STEPS + 1):
                step_count += 1
                fraction_finished: float = (step + 1) / MAX_STEPS
                action: int = self._policy(state)
                observation, reward, done, info = env.step(action)
                observation = append(observation, fraction_finished)
                observation: ndarray = observation[newaxis, ...]
                # env.render()

                if step == MAX_STEPS:
                    done = True

                self._replay_buffer.add(state[0], action, reward, observation[0], done)
                state = observation
                episode_reward += reward

                if self._replay_buffer.size >= TRAINING_START and step_count % TRAIN_FREQUENCY == 0:
                    batch: List[Tuple[ndarray, int, float, ndarray, bool]] = self._replay_buffer.sample_batch(BATCH_SIZE)
                    states, actions, rewards, observations, dones = self._preprocess(batch)
                    q_targets: Tensor = self._compute_q_targets(states, actions, rewards, observations, dones)
                    self._train_step(states, q_targets)

                if done:
                    break

            if episode % BACKUP_FREQUENCY == 0:
                manager.save()

            if episode % REPLACE_FREQUENCY == 0:
                self._update_target_model()

            self._update_epsilon()
            self._update_episode_rewards(episode_reward)
            print("Episode: {} -- Reward: {} -- Average: {}".
                  format(episode, episode_reward, self._average_reward()))

        end: float = time.time()
        print('Time: {}s'.format(end - start))

    @property
    def model(self) -> DDDQN:
        return self._q_model

    @property
    def target_model(self) -> DDDQN:
        return self._target_model


def visualize(network: DDDQN):
    state: ndarray = env.reset()
    for step in range(MAX_STEPS):
        fraction_finished: float = (step + 1) / MAX_STEPS
        state = append(state, fraction_finished)
        state = state[newaxis, ...]
        action: ndarray[int] = argmax(network.advantage(state))
        state, reward, done, info = env.step(action)
        env.render()
        if done:
            break


if __name__ == '__main__':
    # hyperparameters
    BATCH_SIZE: int = 64
    MAX_STEPS: int = 1000
    MAX_EPISODES: int = 1000
    REPLACE_FREQUENCY: int = 50
    BACKUP_FREQUENCY: int = 100
    TRAINING_START: int = 256
    TRAIN_FREQUENCY: int = 4
    EPSILON: float = 0.3
    EPSILON_DECAY_RATE: float = 0.995
    MIN_EPSILON: float = 0.001
    GAMMA: float = 0.999
    LEARNING_RATE: float = 0.001
    REGULARIZATION_FACTOR: float = 0.001

    # set-up environment
    env: gym.Env = gym.make('LunarLander-v2')
    dddqn: DDDQN = DDDQN()
    huber: Loss = Huber()
    adam: Optimizer = Adam(LEARNING_RATE)

    # set-up agent
    agent: Agent = Agent(dddqn)
    checkpoint: tf.train.Checkpoint = tf.train.Checkpoint(q_model=dddqn, optimizer=adam,
                                                          target_model=agent.target_model)
    manager: tf.train.CheckpointManager = tf.train.CheckpointManager(checkpoint, 'lunar_lander/', max_to_keep=3)
    checkpoint.restore(manager.latest_checkpoint)

    # train
    # agent.training()

    # visualize trained agent
    for _ in range(1):
        visualize(agent.model)
