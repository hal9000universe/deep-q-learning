import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
import gym
import os

import time
from typing import Callable, Mapping, Tuple, List
from numpy import ndarray, zeros, float64, int64, argmax, append, newaxis
from jax.nn import one_hot
from numpy.random import uniform, randint

from statistics import mean
from pickle import dump, load


class Model(hk.Module):
    _lin1: hk.Linear
    _lin2: hk.Linear
    _val: hk.Linear
    _adv: hk.Linear

    def __init__(self):
        super().__init__()
        self._lin1 = hk.Linear(64)
        self._lin2 = hk.Linear(64)
        self._val = hk.Linear(1)
        self._adv = hk.Linear(env.action_space.n)

    def __call__(self, x: np.ndarray or jnp.ndarray) -> np.ndarray or jnp.ndarray:
        x = self._lin1(x)
        x = jax.nn.relu(x)
        x = self._lin2(x)
        x = jax.nn.relu(x)
        val: np.ndarray or jnp.ndarray = self._val(x)
        adv: np.ndarray or jnp.ndarray = self._adv(x)
        Q: np.ndarray or jnp.ndarray = val + adv - jnp.mean(adv, axis=1, keepdims=True)
        return Q


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
                 obs_shape: Tuple[int, int],
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
    pred: jnp.ndarray = model.apply(params, inp)
    loss_val: jnp.ndarray = jnp.mean(jnp.sum(optax.huber_loss(pred, targ), axis=1), axis=0)
    return loss_val


@jax.jit
def compute_q_targets(params: hk.Params, target_params: hk.Params,
                      states: ndarray, actions: ndarray, rewards: ndarray,
                      observations: ndarray, dones: ndarray) -> jnp.ndarray:
    q: ndarray = model.apply(params, states)
    next_q: ndarray = model.apply(params, observations)
    next_q_tm: ndarray = model.apply(target_params, observations)
    max_actions: ndarray = argmax(next_q, axis=1)
    targets: List = []
    for index, action in enumerate(max_actions):
        target_val: float = rewards[index] + (GAMMA * next_q_tm[index, action] - q[index, actions[index]]) * \
                            (1.0 - dones[index])

        q_target: ndarray = q[index] + target_val * one_hot(actions[index], env.action_space.n)
        targets.append(q_target)
    targets: jnp.ndarray = jnp.array(targets)
    return targets


class Agent:
    _replay_buffer: ReplayBuffer
    _q_model: hk.Transformed
    _model_version: int
    _epsilon: float
    _episode_rewards: List[float]

    def __init__(self, q_net, params: hk.Params, opt: optax.adam,
                 opt_state: Mapping, loss_fn: Callable):
        buffer_size: int = 100000
        self._replay_buffer = ReplayBuffer(buffer_size=buffer_size, obs_shape=(buffer_size, 9))
        self._q_model = q_net
        self._params = params
        self._opt_state = opt_state
        self._optimizer = opt
        self._loss = loss_fn
        self._target_params = params
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

    def _policy(self, x: ndarray) -> int or ndarray:
        if self._epsilon < uniform(0, 1):
            action: ndarray = argmax(model.apply(self._params, x))
            return int(action)
        else:
            return randint(0, 4)

    def _update_target_model(self):
        self._target_params = self._params

    def _save(self):
        if not os.path.exists("lunar_lander"):
            os.mkdir("lunar_lander")
        with open("lunar_lander/params.pickle", "wb") as f:
            dump(self._params, f)
        with open("lunar_lander/opt_state.pickle", "wb") as f:
            dump(self._opt_state, f)

    def _load(self):
        with open("lunar_lander/params.pickle", "rb") as f:
            self._params = load(f)
        with open("lunar_lander/opt_state.pickle", "rb") as f:
            self._opt_state = load(f)

    def training(self):
        step_count: int = 0
        for episode in range(MAX_EPISODES):
            start: float = time.time()
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
                self._save()

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
    BACKUP_FREQUENCY: int = 5
    TRAINING_START: int = 256
    TRAIN_FREQUENCY: int = 4
    EPSILON: float = 1.0
    EPSILON_DECAY_RATE: float = 0.995
    MIN_EPSILON: float = 0.001
    GAMMA: float = 0.999
    LEARNING_RATE: float = 0.001

    env: gym.Env = gym.make('LunarLander-v2')

    rng: jax.random.PRNGKeyArray = jax.random.PRNGKey(time.time_ns())
    test_input: np.ndarray = zeros((1, 9))

    model: hk.Transformed = hk.without_apply_rng(hk.transform(lambda *args: Model()(*args)))
    optimizer: optax.adam = optax.adam(LEARNING_RATE)

    parameters: hk.Params = model.init(rng, test_input)
    optimizer_state: Mapping = optimizer.init(parameters)

    agent = Agent(model, parameters, optimizer, optimizer_state, compute_loss)
    agent.training()
