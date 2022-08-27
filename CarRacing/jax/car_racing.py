from abc import ABC

import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
import gym
import os

import time
from typing import Mapping, Tuple, List
from numpy import ndarray, zeros, float64, int64, argmax, newaxis, array
from jax.nn import one_hot
from numpy.random import uniform, randint

from statistics import mean
from pickle import dump, load


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
    environment: gym.Env = gym.make('CarRacing-v1')
    environment = DiscreteActionWrapper(environment, cont_ac_list)
    environment = gym.wrappers.FrameStack(environment, 4)
    environment = gym.wrappers.NormalizeObservation(environment)
    environment = gym.wrappers.TransformObservation(environment, transform_observation)
    return environment


class Model(hk.Module):
    _conv1: hk.Conv3D
    _conv2: hk.Conv3D
    _batch_norm1: hk.BatchNorm
    _batch_norm2: hk.BatchNorm
    _flatten: hk.Flatten
    _lin1: hk.Linear
    _q: hk.Linear

    def __init__(self):
        super().__init__()
        self._relu = jax.nn.relu
        self._conv1 = hk.Conv3D(128, (1, 2, 2))
        self._conv2 = hk.Conv3D(64, (2, 3, 3))
        self._flatten = hk.Flatten()
        self._lin1 = hk.Linear(64)
        self._q = hk.Linear(env.action_space.n)

    def __call__(self, x: ndarray or jnp.ndarray, training: bool = False) -> ndarray or jnp.ndarray:
        x = self._conv1(x)
        x = self._relu(x)
        x = self._conv2(x)
        x = self._relu(x)
        x = self._flatten(x)
        x = self._lin1(x)
        x = self._relu(x)
        x = self._q(x)
        x = self._relu(x)
        return x


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
        self._states = zeros(obs_shape, dtype=float64)
        self._actions = zeros((buffer_size,), dtype=int64)
        self._rewards = zeros((buffer_size,), dtype=float64)
        self._observations = zeros(obs_shape, dtype=float64)
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
        random_indices: ndarray = randint(0, self._samples, batch_size)
        batch = (self._states[random_indices], self._actions[random_indices], self._rewards[random_indices],
                 self._observations[random_indices], self._dones[random_indices])
        return batch


@jax.jit
def train_step(params: hk.Params, opt_state: Mapping,
               states: jnp.ndarray, q_targets: jnp.ndarray) -> Tuple[hk.Params, Mapping]:
    grads = jax.grad(compute_loss)(params, states, q_targets)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state


def compute_loss(params: hk.Params, inp: jnp.ndarray, targ: jnp.ndarray) -> jnp.ndarray:
    pred: jnp.ndarray = model.apply(params, inp, True)
    loss_val: jnp.ndarray = jnp.mean(jnp.sum(optax.huber_loss(pred, targ), axis=1), axis=0)
    return loss_val


@jax.jit
def compute_q_targets(params: hk.Params, target_params: hk.Params,
                      states: jnp.ndarray, actions: ndarray, rewards: ndarray,
                      observations: ndarray, dones: ndarray) -> jnp.ndarray:
    q: ndarray = model.apply(params, states, True)
    next_q: ndarray = model.apply(params, observations, True)
    next_q_tm: ndarray = model.apply(target_params, observations, True)
    max_actions: ndarray = argmax(next_q, axis=1)
    targets: List = []
    for index, action in enumerate(max_actions):
        target_val: float = rewards[index] + \
                            (1.0 - dones[index]) * (GAMMA * next_q_tm[index, action] - q[index, actions[index]])

        q_target: ndarray = q[index] + target_val * one_hot(actions[index], env.action_space.n)
        targets.append(q_target)
    targets: jnp.ndarray = jnp.array(targets)
    return targets


class Agent:
    _replay_buffer: ReplayBuffer
    _params: hk.Params
    _opt_state: Mapping
    _epsilon: float
    _episode_rewards: List[float]

    def __init__(self, params: hk.Params, opt_state: Mapping):
        buffer_size: int = 1000
        self._replay_buffer = ReplayBuffer(buffer_size=buffer_size, obs_shape=(buffer_size, 4, 96, 96, 3))
        self._params = params
        self._opt_state = opt_state
        self._target_params = params
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

    def _policy(self, x: ndarray) -> int or ndarray:
        if self._epsilon < uniform(0, 1):
            action: ndarray = argmax(model.apply(self._params, x))
            return int(action)
        else:
            return randint(0, 4)

    def _update_target_model(self):
        self._target_params = self._params

    def save(self):
        if not os.path.exists("car_racing"):
            os.mkdir("car_racing")
        with open("car_racing/params.pickle", "wb") as file:
            dump(self._params, file)

    def load(self):
        with open("car_racing/params.pickle", "rb") as file:
            self._params = load(file)

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
        step_count: int = 0
        for episode in range(MAX_EPISODES):
            start: float = time.time()
            episode_reward: float = 0.
            non_positive_counter: int = 0
            state: ndarray = env.reset()
            for step in range(1, MAX_STEPS + 1):
                step_count += 1
                action: int = self._policy(state)
                observation, reward, done, info = env.step(action)

                if step == MAX_STEPS:
                    done: bool = True

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
                    states, actions, rewards, observations, dones = self._replay_buffer.sample_batch(BATCH_SIZE)
                    states: jnp.ndarray = jax.numpy.asarray(states)
                    dones = dones.astype(float64)
                    q_targets: jnp.ndarray = compute_q_targets(self._params, self._target_params, states,
                                                               actions, rewards, observations, dones)
                    self._params, self._opt_state = train_step(self._params, self._opt_state,
                                                               states, q_targets)

                if done:
                    break

            if episode % REPLACE_FREQUENCY == 0:
                self._update_target_model()

            if episode % BACKUP_FREQUENCY == 0:
                self.save()

            self._update_epsilon()
            self._update_episode_rewards(episode_reward)
            print("Episode: {} -- Reward: {} -- Average: {}".
                  format(episode, episode_reward, self._average_reward()))

            end: float = time.time()
            print('Time: {}s'.format(end - start))


if __name__ == '__main__':
    BATCH_SIZE: int = 64
    MAX_STEPS: int = 1000
    MAX_EPISODES: int = 10000
    REPLACE_FREQUENCY: int = 50
    BACKUP_FREQUENCY: int = 50
    TRAINING_START: int = 256
    TRAIN_FREQUENCY: int = 4
    EPSILON: float = 1.0
    EPSILON_DECAY_RATE: float = 0.995
    MIN_EPSILON: float = 0.001
    GAMMA: float = 0.999
    LEARNING_RATE: float = 0.001
    MAX_NON_POSITIVE: int = 50

    cont_ac_list: List[ndarray] = [array([0, 1, 0]), array([1, 1, 0]), array([-1, 1, 0]), array([0.5, 1, 0]),
                                   array([-0.5, 1, 0]), array([1, 0.1, 0]), array([-1, 0.1, 0]), array([0, 0, 0.8]),
                                   array([1, 0, 0.5]), array([-1, 0, 0.5]), array([0, 0.5, 0]), array([0.5, 0.5, 0]),
                                   array([-0.5, 0.5, 0])]
    env: gym.Env = create_environment()

    rng: jax.random.PRNGKeyArray = jax.random.PRNGKey(time.time_ns())
    test_input: np.ndarray = env.reset()

    model: hk.Transformed = hk.without_apply_rng(hk.transform(lambda *args: Model()(*args)))
    optimizer: optax.adam = optax.adam(LEARNING_RATE)

    parameters: hk.Params = model.init(rng, test_input)
    optimizer_state: Mapping = optimizer.init(parameters)

    agent = Agent(parameters, optimizer_state)
    agent.training()
