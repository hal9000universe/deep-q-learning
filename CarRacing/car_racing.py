from abc import ABC
from typing import List, Tuple, Optional

import gym
import time

import tensorflow as tf
from tensorflow import Tensor, convert_to_tensor, cast, float64, one_hot, GradientTape, function
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv3D, BatchNormalization, MaxPooling3D, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Optimizer, Adam
from tensorflow.keras.losses import Loss, Huber

from numpy import ndarray, array, newaxis, argmax
from statistics import mean
from random import uniform, sample, randint


class DDDQN(Model, ABC):
    _conv1: Conv3D
    _conv2: Conv3D
    _batch_norm1: BatchNormalization
    _batch_norm2: BatchNormalization
    _max_pool1: MaxPooling3D
    _max_pool2: MaxPooling3D
    _flatten: Flatten
    _dense1: Dense
    _dense2: Dense
    _val: Dense
    _adv: Dense

    def __init__(self):
        super(DDDQN, self).__init__()
        self._conv1 = Conv3D(128, (1, 2, 2), activation='relu', kernel_regularizer=l2(REGULARIZATION_FACTOR),
                             input_shape=(4, 96, 96, 3))
        self._conv2 = Conv3D(64, (2, 3, 3), activation='relu', kernel_regularizer=l2(REGULARIZATION_FACTOR))
        self._batch_norm1 = BatchNormalization()
        self._batch_norm2 = BatchNormalization()
        self._max_pool1 = MaxPooling3D((1, 2, 2))
        self._max_pool2 = MaxPooling3D((1, 2, 2))
        self._flatten = Flatten()
        self._dense1 = Dense(128, activation='relu', kernel_regularizer=l2(REGULARIZATION_FACTOR))
        self._dense2 = Dense(64, activation='relu', kernel_regularizer=l2(REGULARIZATION_FACTOR))
        self._val = Dense(1, activation='linear')
        self._adv = Dense(env.action_space.n, activation='linear')

    def call(self, x: ndarray, training: bool = False, mask=None) -> Tensor:
        x = self._conv1(x)
        x = self._max_pool1(x)
        x = self._batch_norm1(x)
        x = self._conv2(x)
        x = self._max_pool2(x)
        x = self._batch_norm2(x)
        x = self._flatten(x)
        x = self._dense1(x)
        x = self._dense2(x)
        val = self._val(x)
        adv = self._adv(x)
        Q = val + adv - tf.math.reduce_mean(adv, axis=1, keepdims=True)
        return Q

    def advantage(self, x: ndarray) -> Tensor:
        x = self._conv1(x)
        x = self._max_pool1(x)
        x = self._batch_norm1(x)
        x = self._conv2(x)
        x = self._max_pool2(x)
        x = self._batch_norm2(x)
        x = self._flatten(x)
        x = self._dense1(x)
        x = self._dense2(x)
        adv = self._adv(x)
        return adv


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
    _memory: List[Optional[Tuple[ndarray, int, float, ndarray, bool]]]
    _counter: int
    _samples: int

    def __init__(self, buffer_size: int = 1000000):
        self._buffer_size = buffer_size
        self._memory = [None] * self._buffer_size
        self._counter = 0
        self._samples = 0

    @property
    def size(self) -> int:
        return self._samples

    def add(self, state: ndarray, action: int, reward: float, observation: ndarray, done: bool):
        experience: Tuple[ndarray, int, float, ndarray, bool] = (state, action, reward, observation, done)
        self._memory[self._counter % self._buffer_size] = experience
        self._counter += 1
        self._samples = min(self._counter, self._buffer_size)

    def sample_batch(self, batch_size: int = 64) -> List[Tuple[ndarray, int, float, ndarray, bool]]:
        batch: List[Tuple[ndarray, int, float, ndarray, bool]] = sample(self._memory[0:self._samples], batch_size)
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
        self._q_model(test_input)
        self._target_model(test_input)

    def training(self):
        self._build()
        start: float = time.time()
        step_count: int = 0
        for episode in range(MAX_EPISODES):
            print('Starting episode ...')
            episode_reward: float = 0.
            state: ndarray = env.reset()
            for step in range(1, MAX_STEPS + 1):
                step_count += 1
                action: int = self._policy(state)
                observation, reward, done, info = env.step(action)
                env.render()

                if step == MAX_STEPS:
                    done = True

                self._replay_buffer.add(state[0], action, reward, observation[0], done)
                state = observation
                episode_reward += reward

                if self._replay_buffer.size >= TRAINING_START and step_count % TRAIN_FREQUENCY == 0:
                    print('Training step: {}'.format(step_count))
                    sampling_start: float = time.time()
                    batch: List[Tuple[ndarray, int, float, ndarray, bool]] = self._replay_buffer.sample_batch(
                        BATCH_SIZE)
                    sampling_end: float = time.time()
                    print('Sampling time: {}'.format(sampling_end - sampling_start))
                    preprocessing_start: float = time.time()
                    states, actions, rewards, observations, dones = self._preprocess(batch)
                    preprocessing_end: float = time.time()
                    print('Preprocessing time: {}'.format(preprocessing_end - preprocessing_start))
                    q_start: float = time.time()
                    q_targets: Tensor = self._compute_q_targets(states, actions, rewards, observations, dones)
                    q_end: float = time.time()
                    print('Q-time: {}'.format(q_end - q_start))
                    train_start: float = time.time()
                    self._train_step(states, q_targets)
                    train_end: float = time.time()
                    print('Training time: {}'.format(train_end - train_start))

                if done:
                    break

                if step % BACKUP_FREQUENCY == 0:
                    print('Saving model ...')
                    manager.save()

                if step % REPLACE_FREQUENCY == 0:
                    print('Updating target model ...')
                    self._update_target_model()

            self._update_epsilon()
            self._update_episode_rewards(episode_reward)
            print('Episode: {} -- Reward: {} -- Average: {}'.
                  format(episode, episode_reward, self._average_reward()))

        end: float = time.time()
        print('Time: {}s'.format(end - start))

    @property
    def model(self) -> DDDQN:
        return self._q_model

    @property
    def target_model(self) -> DDDQN:
        return self._target_model


def visualize(model: DDDQN):
    state: ndarray = env.reset()
    for _ in range(MAX_STEPS):
        action = argmax(model.advantage(state))
        state, reward, done, info = env.step(action)
        print(reward)
        env.render()
        if done:
            break


if __name__ == '__main__':
    # hyperparameters
    BATCH_SIZE: int = 64
    MAX_STEPS: int = 1000
    MAX_EPISODES: int = 1000
    REPLACE_FREQUENCY: int = 200
    BACKUP_FREQUENCY: int = 250
    TRAINING_START: int = 256
    TRAIN_FREQUENCY: int = 10
    EPSILON: float = 0.8
    EPSILON_DECAY_RATE: float = 0.995
    MIN_EPSILON: float = 0.001
    GAMMA: float = 0.999
    LEARNING_RATE: float = 0.001
    REGULARIZATION_FACTOR: float = 0.001

    # set-up environment
    cont_ac_list: List[ndarray] = [array([0, 1, 0]), array([1, 1, 0]), array([-1, 1, 0]), array([0.5, 1, 0]),
                                   array([-0.5, 1, 0]), array([1, 0.1, 0]), array([-1, 0.1, 0]), array([0, 0, 0.8]),
                                   array([1, 0, 0.5]), array([-1, 0, 0.5]), array([0, 0.5, 0]), array([0.5, 0.5, 0]),
                                   array([-0.5, 0.5, 0])]
    env: gym.Env = create_environment()

    dddqn: DDDQN = DDDQN()
    huber: Loss = Huber()
    adam: Optimizer = Adam(LEARNING_RATE)

    # set-up agent
    agent: Agent = Agent(dddqn)
    checkpoint: tf.train.Checkpoint = tf.train.Checkpoint(q_model=dddqn, optimizer=adam,
                                                          target_model=agent.target_model)
    manager: tf.train.CheckpointManager = tf.train.CheckpointManager(checkpoint, 'car_racing/', max_to_keep=3)
    # checkpoint.restore(manager.latest_checkpoint)

    visualize(agent.model)

    agent.training()


# TODO: optimize replay buffer structure
# TODO: implement early episode stopping mechanism