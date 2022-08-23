from abc import ABC
from typing import List, Tuple

import gym
import time

import tensorflow as tf
from tensorflow import Tensor, convert_to_tensor, cast, float64, one_hot, GradientTape, function
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv3D, BatchNormalization, MaxPooling3D, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Optimizer, Adam
from tensorflow.keras.losses import Loss, Huber

from numpy import ndarray, array, newaxis, argmax, random, zeros
from statistics import mean
from random import uniform, randint


class DDQN(Model, ABC):
    _conv1: Conv3D
    _conv2: Conv3D
    _batch_norm1: BatchNormalization
    _batch_norm2: BatchNormalization
    _max_pool1: MaxPooling3D
    _max_pool2: MaxPooling3D
    _flatten: Flatten
    _dense: Dense
    _q: Dense

    def __init__(self):
        super(DDQN, self).__init__()
        self._conv1 = Conv3D(128, (1, 2, 2), activation='relu', kernel_regularizer=l2(REGULARIZATION_FACTOR),
                             input_shape=(4, 96, 96, 3))
        self._conv2 = Conv3D(64, (2, 3, 3), activation='relu', kernel_regularizer=l2(REGULARIZATION_FACTOR))
        self._batch_norm1 = BatchNormalization()
        self._batch_norm2 = BatchNormalization()
        self._max_pool1 = MaxPooling3D((1, 2, 2))
        self._max_pool2 = MaxPooling3D((1, 2, 2))
        self._flatten = Flatten()
        self._dense = Dense(64, activation='relu', kernel_regularizer=l2(REGULARIZATION_FACTOR))
        self._q = Dense(env.action_space.n, activation='linear')

    def call(self, x: ndarray, training: bool = False, mask=None) -> Tensor:
        x = self._conv1(x)
        x = self._max_pool1(x)
        x = self._batch_norm1(x)
        x = self._conv2(x)
        x = self._max_pool2(x)
        x = self._batch_norm2(x)
        x = self._flatten(x)
        x = self._dense(x)
        Q = self._q(x)
        return Q


class DiscreteActionWrapper(gym.ActionWrapper, ABC):
    _disc_to_cont: List[ndarray]
    _action_space: gym.spaces.Discrete

    def __init__(self, environment: gym.Env, disc_to_cont: List[ndarray]):
        super(DiscreteActionWrapper, self).__init__(environment)
        self._disc_to_cont = disc_to_cont
        self._action_space = gym.spaces.Discrete(len(disc_to_cont))

    def action(self, act):
        return self._disc_to_cont[act]


def transform_observation(obs: ndarray) -> ndarray:
    obs = obs[newaxis, ...]
    return obs


def create_environment() -> gym.Env:
    environment: gym.Env = gym.make('CarRacing-v0')
    environment = DiscreteActionWrapper(environment, cont_ac_list)
    environment = gym.wrappers.FrameStack(environment, 4)
    environment = gym.wrappers.NormalizeObservation(environment)
    environment = gym.wrappers.TransformObservation(environment, transform_observation)
    return environment


class ReplayBuffer:
    _buffer_size: int
    _states: ndarray
    _actions: ndarray
    _rewards: ndarray
    _observations: ndarray
    _dones: ndarray
    _counter: int
    _samples: int

    def __init__(self,
                 obs_shape: Tuple,
                 buffer_size: int = 1000000):
        self._buffer_size = buffer_size
        self._states = zeros(obs_shape, dtype=float)
        self._actions = zeros((buffer_size,), dtype=int)
        self._rewards = zeros((buffer_size,), dtype=float)
        self._observations = zeros(obs_shape, dtype=float)
        self._dones = zeros((buffer_size,), dtype=bool)
        self._counter = 0
        self._samples = 0

    @property
    def size(self) -> int:
        return self._samples

    def add(self, state: ndarray, action: int, reward: float, observation: ndarray, done: bool):
        self._states[self._counter % self._buffer_size] = state
        self._actions[self._counter % self._buffer_size] = action
        self._rewards[self._counter % self._buffer_size] = reward
        self._observations[self._counter % self._buffer_size] = observation
        self._dones[self._counter % self._buffer_size] = done
        self._counter += 1
        self._samples = min(self._counter, self._buffer_size)

    def sample_batch(self, batch_size: int = 64) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
        random_indices: ndarray = random.randint(0, self._samples - 1, batch_size)
        batch = (self._states[random_indices], self._actions[random_indices], self._rewards[random_indices],
                 self._observations[random_indices], self._dones[random_indices])
        return batch


class Agent:
    _replay_buffer: ReplayBuffer
    _q_model: DDQN
    _target_model: DDQN
    _model_version: int
    _epsilon: float
    _episode_rewards: List[float]

    def __init__(self, q_net: DDQN):
        buffer_size: int = 100000
        self._replay_buffer = ReplayBuffer(buffer_size=buffer_size, obs_shape=(buffer_size, 4, 96, 96, 3))
        self._q_model = q_net
        self._target_model = DDQN()
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
            action: ndarray[int] = argmax(self._q_model(x))
            return action
        else:
            action: int = randint(0, 3)
            return action

    def _update_target_model(self):
        self._target_model.set_weights(self._q_model.get_weights())

    def _compute_q_targets(self, states: Tensor, actions: ndarray, rewards: ndarray, observations: Tensor,
                           dones: ndarray) -> Tensor:
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
            q_target: Tensor = cast(q[index], dtype=float64) + one_hot(actions[index], env.action_space.n,
                                                                       on_value=cast(target_val, dtype=float64))
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
    def _preprocess(batch: Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]
                    ) -> Tuple[Tensor, ndarray, ndarray, Tensor, ndarray]:
        states: Tensor = convert_to_tensor(batch[0], dtype=tf.float64)
        observations: Tensor = convert_to_tensor(batch[3], dtype=tf.float64)
        return states, batch[1], batch[2], observations, batch[4]

    def _build(self):
        test_input: ndarray = env.reset()
        self._q_model(test_input)
        self._target_model(test_input)

    @staticmethod
    def _check_early_stop(non_positive_counter: int) -> Tuple[bool, float]:
        done: bool = (non_positive_counter > MAX_NON_POSITIVE)
        if done:
            print('Stopping early')
            punishment: float = -80
            return done, punishment
        else:
            return False, 0.0

    def training(self):
        self._build()
        start: float = time.time()
        step_count: int = 0
        for episode in range(MAX_EPISODES):
            print('Starting episode ...')
            episode_reward: float = 0.
            non_positive_counter: int = 0
            state: ndarray = env.reset()
            for step in range(1, MAX_STEPS + 1):
                step_count += 1
                action: int = self._policy(state)
                observation, reward, done, info = env.step(action)
                env.render()

                print(reward)

                if step == MAX_STEPS:
                    done = True

                if reward <= 0:
                    non_positive_counter += 1
                else:
                    non_positive_counter = 0

                early_stop, punishment = self._check_early_stop(non_positive_counter)
                if early_stop:
                    done = early_stop
                    reward += punishment

                self._replay_buffer.add(state[0], action, reward, observation[0], done)
                state = observation
                episode_reward += reward

                if self._replay_buffer.size >= TRAINING_START and step_count % TRAIN_FREQUENCY == 0:
                    batch: Tuple[ndarray, ndarray, ndarray, ndarray, ndarray] = self._replay_buffer.sample_batch(
                        BATCH_SIZE)
                    states, actions, rewards, observations, dones = self._preprocess(batch)
                    q_targets: Tensor = self._compute_q_targets(states, actions, rewards, observations, dones)
                    self._train_step(states, q_targets)

                if done:
                    break

                if step_count % BACKUP_FREQUENCY == 0:
                    print('Saving model ...')
                    manager.save()

            if episode % REPLACE_FREQUENCY == 0:
                print('Updating target model ...')
                self._update_target_model()

            self._update_epsilon()
            self._update_episode_rewards(episode_reward)
            print('Episode: {} -- Reward: {} -- Average: {}'.
                  format(episode, episode_reward, self._average_reward()))

        end: float = time.time()
        print('Time: {}s'.format(end - start))

    @property
    def model(self) -> DDQN:
        return self._q_model

    @property
    def target_model(self) -> DDQN:
        return self._target_model


def visualize(model: DDQN):
    state: ndarray = env.reset()
    for _ in range(MAX_STEPS):
        action = argmax(model(state))
        state, reward, done, info = env.step(action)
        print(reward)
        env.render()
        if done:
            break


if __name__ == '__main__':
    # hyperparameters
    BATCH_SIZE: int = 64
    MAX_STEPS: int = 50000
    MAX_EPISODES: int = 1000
    REPLACE_FREQUENCY: int = 10000
    BACKUP_FREQUENCY: int = 100
    TRAINING_START: int = 256
    TRAIN_FREQUENCY: int = 1
    EPSILON: float = 1.0
    EPSILON_DECAY_RATE: float = 0.995
    MIN_EPSILON: float = 0.02
    GAMMA: float = 0.999
    LEARNING_RATE: float = 0.001
    REGULARIZATION_FACTOR: float = 0.001
    MAX_NON_POSITIVE: int = 50

    # set-up environment
    cont_ac_list: List[ndarray] = [array([0, 1, 0]), array([1, 1, 0]), array([-1, 1, 0]), array([0.5, 1, 0]),
                                   array([-0.5, 1, 0]), array([1, 0.1, 0]), array([-1, 0.1, 0]), array([0, 0, 0.8]),
                                   array([1, 0, 0.5]), array([-1, 0, 0.5]), array([0, 0.5, 0]), array([0.5, 0.5, 0]),
                                   array([-0.5, 0.5, 0])]
    env: gym.Env = create_environment()

    dddqn: DDQN = DDQN()
    huber: Loss = Huber()
    adam: Optimizer = Adam(LEARNING_RATE)

    # set-up agent
    agent: Agent = Agent(dddqn)
    checkpoint: tf.train.Checkpoint = tf.train.Checkpoint(q_model=dddqn, optimizer=adam,
                                                          target_model=agent.target_model)
    manager: tf.train.CheckpointManager = tf.train.CheckpointManager(checkpoint, 'car_racing/', max_to_keep=3)
    checkpoint.restore(manager.latest_checkpoint)

    agent.training()
