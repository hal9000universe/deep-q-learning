# py
import os
import time
import numba
import asyncio
from statistics import mean
from pickle import dump, load
from typing import Mapping, Tuple, List, Any, Callable

# nn & rl
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
import gym
from jax.nn import one_hot
from numpy import ndarray, zeros, float64, int64, argmax, append, newaxis
from numpy.random import uniform, randint


class ObsWrapper(gym.Wrapper):

    def __init__(self, environment):
        super(ObsWrapper, self).__init__(environment)
        self._step = 0
        self._observation_space = gym.spaces.Box(shape=(1, 9), low=float('-inf'), high=float('inf'))

    def observation(self, observation: ndarray) -> ndarray:
        fraction_finished: float = self._step / MAX_STEPS
        return append(observation, fraction_finished)[newaxis, ...]

    def step(self, action) -> Tuple[ndarray, float, bool, Any]:
        self._step += 1
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def reset(self, **kwargs) -> ndarray:
        self._step = 0
        state: ndarray = self.env.reset()
        return self.observation(state)


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

    def __call__(self, x: ndarray or jnp.ndarray) -> ndarray or jnp.ndarray:
        x = self._lin1(x)
        x = jax.nn.relu(x)
        x = self._lin2(x)
        x = jax.nn.relu(x)
        val: ndarray or jnp.ndarray = self._val(x)
        adv: ndarray or jnp.ndarray = self._adv(x)
        Q: ndarray or jnp.ndarray = val + adv - jnp.mean(adv, axis=1, keepdims=True)
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

    @property
    def states(self) -> ndarray:
        return self._states

    @property
    def actions(self) -> ndarray:
        return self._actions

    @property
    def rewards(self) -> ndarray:
        return self._rewards

    @property
    def observations(self) -> ndarray:
        return self._observations

    @property
    def dones(self) -> ndarray:
        return self._dones

    def add(self, state: ndarray, action: int, reward: float, observation: ndarray, done: bool):
        self._states[self._counter % self._buffer_size] = state
        self._actions[self._counter % self._buffer_size] = action
        self._rewards[self._counter % self._buffer_size] = reward
        self._observations[self._counter % self._buffer_size] = observation
        self._dones[self._counter % self._buffer_size] = done
        self._counter += 1
        self._samples = min(self._counter, self._buffer_size)


@numba.njit
def sample_batch(num_samples: int, states: ndarray, actions: ndarray, rewards: ndarray,
                 observations: ndarray, dones: ndarray, batch_size: int = 64) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
    random_indices: ndarray = randint(0, num_samples, batch_size)
    batch = (states[random_indices], actions[random_indices], rewards[random_indices],
             observations[random_indices], dones[random_indices])
    return batch


@jax.jit
def train_step(params: hk.Params, opt_state: Mapping,
               states: jnp.ndarray, q_targets: jnp.ndarray) -> Tuple[hk.Params, Mapping]:
    grads = jax.grad(compute_loss)(params, states, q_targets)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state


def compute_loss(params: hk.Params, states: jnp.ndarray, q_targets: jnp.ndarray) -> jnp.ndarray:
    pred: jnp.ndarray = model.apply(params, states)
    loss_val: jnp.ndarray = jnp.mean(jnp.sum(optax.huber_loss(pred, q_targets), axis=1), axis=0)
    return loss_val


@jax.jit
def compute_q_targets(params: hk.Params, target_params: hk.Params,
                      states: jnp.ndarray, actions: ndarray, rewards: ndarray,
                      observations: ndarray, dones: ndarray) -> jnp.ndarray:
    q: ndarray = model.apply(params, states)
    next_q: ndarray = model.apply(params, observations)
    next_q_tm: ndarray = model.apply(target_params, observations)
    max_actions: ndarray = argmax(next_q, axis=1)
    targets: List = []
    for index, (max_action, action, done) in enumerate(zip(max_actions, actions, dones)):
        target_val: float = rewards[index] + (1.0 - done) * (GAMMA * next_q_tm[index, max_action] - q[index, action])
        q_target: ndarray = q[index] + target_val * one_hot(action, env.action_space.n)
        targets.append(q_target)
    targets: jnp.ndarray = jnp.array(targets)
    return targets


def generate_action_computation(network: hk.Transformed) -> Callable:
    @jax.jit
    def action_computation(params: hk.Params, state: ndarray) -> ndarray:
        action: ndarray = argmax(network.apply(params, state))
        return action
    return action_computation


@jax.jit
def preprocessing(states: ndarray, actions: ndarray, rewards: ndarray, observations: ndarray,
                  dones: ndarray) -> Tuple[jnp.ndarray, ndarray, ndarray, ndarray, ndarray]:
    states: jnp.ndarray = jax.numpy.asarray(states)
    dones = dones.astype(float64)
    return states, actions, rewards, observations, dones


class Agent:
    _replay_buffer: ReplayBuffer
    _params: hk.Params
    _opt_state: Mapping
    _epsilon: float
    _episode_rewards: List[float]

    def __init__(self, params: hk.Params, opt_state: Mapping):
        self._replay_buffer = ReplayBuffer(buffer_size=BUFFER_SIZE, obs_shape=(BUFFER_SIZE, 9))
        self._params = params
        self._opt_state = opt_state
        self._target_params = params
        self._epsilon = EPSILON
        self._episode_rewards = []

    async def _update_epsilon(self):
        self._epsilon = max(self._epsilon * EPSILON_DECAY_RATE, MIN_EPSILON)

    async def _update_episode_rewards(self, episode_reward: float):
        self._episode_rewards.append(episode_reward)
        while len(self._episode_rewards) > 50:
            self._episode_rewards.pop(0)

    def _average_reward(self) -> float:
        return mean(self._episode_rewards)

    def _policy(self, state: ndarray) -> int:
        if self._epsilon < uniform(0, 1):
            return int(compute_action(self._params, state))
        else:
            return randint(0, 4)

    async def _update_target_model(self):
        self._target_params = self._params

    def training(self):
        step_count: int = 0
        for episode in range(MAX_EPISODES):
            start: float = time.time()
            epi_reward: float = 0.
            state: ndarray = env.reset()
            for step in range(1, MAX_STEPS + 1):
                step_count += 1
                action: int = self._policy(state)
                observation, reward, done, info = env.step(action)

                if step == MAX_STEPS:
                    done: bool = True

                self._replay_buffer.add(state[0], action, reward, observation[0], done)
                state = observation
                epi_reward += reward

                if self._replay_buffer.size >= TRAINING_START and step_count % TRAIN_FREQUENCY == 0:
                    states, actions, rewards, observations, dones = sample_batch(self._replay_buffer.size,
                                                                                 self._replay_buffer.states,
                                                                                 self._replay_buffer.actions,
                                                                                 self._replay_buffer.rewards,
                                                                                 self._replay_buffer.observations,
                                                                                 self._replay_buffer.dones,
                                                                                 BATCH_SIZE)
                    states, actions, rewards, observations, dones = preprocessing(states,
                                                                                  actions,
                                                                                  rewards,
                                                                                  observations,
                                                                                  dones)
                    q_targets: jnp.ndarray = compute_q_targets(self._params,
                                                               self._target_params,
                                                               states,
                                                               actions,
                                                               rewards,
                                                               observations,
                                                               dones)
                    self._params, self._opt_state = train_step(self._params,
                                                               self._opt_state,
                                                               states,
                                                               q_targets)

                if done:
                    break

            if episode % REPLACE_FREQUENCY == 0:
                asyncio.run(self._update_target_model())

            # if episode % BACKUP_FREQUENCY == 0:
            #     asyncio.run(save_training_state(self._params))

            asyncio.run(self._update_epsilon())
            asyncio.run(self._update_episode_rewards(epi_reward))

            if self._average_reward() > 240:
                asyncio.run(save_training_state(self._params))
                return

            end: float = time.time()
            if episode % 10 == 0:
                print("Episode: {} -- Reward: {} -- Average: {}".format(episode, epi_reward, self._average_reward()))
                print('Time: {}s'.format(end - start))


async def save_training_state(params: hk.Params):
    if not os.path.exists("lunar_lander"):
        os.mkdir("lunar_lander")
    with open("lunar_lander/params.pickle", "wb") as file:
        dump(params, file)


def load_state() -> hk.Params:
    with open("lunar_lander/params.pickle", "rb") as file:
        params: hk.Params = load(file)
    return params


def visualize_agent():
    state: ndarray = env.reset()
    for step in range(MAX_STEPS):
        action: int = int(argmax(model.apply(parameters, state)))
        state, reward, done, info = env.step(action)
        env.render()


if __name__ == '__main__':
    BATCH_SIZE: int = 64
    BUFFER_SIZE: int = 100000
    MAX_STEPS: int = 1000
    MAX_EPISODES: int = 10000
    REPLACE_FREQUENCY: int = 50
    BACKUP_FREQUENCY: int = 20
    TRAINING_START: int = 256
    TRAIN_FREQUENCY: int = 4
    EPSILON: float = 1.0
    EPSILON_DECAY_RATE: float = 0.995
    MIN_EPSILON: float = 0.001
    GAMMA: float = 0.999
    LEARNING_RATE: float = 0.001

    env: gym.Env = ObsWrapper(gym.make('LunarLander-v2'))

    rng: jax.random.PRNGKeyArray = jax.random.PRNGKey(time.time_ns())
    test_input: np.ndarray = zeros((1, 9))

    model: hk.Transformed = hk.without_apply_rng(hk.transform(lambda *args: Model()(*args)))
    optimizer: optax.adam = optax.adam(LEARNING_RATE)

    parameters: hk.Params = model.init(rng, test_input)
    optimizer_state = optimizer.init(parameters)

    compute_action: Callable = generate_action_computation(model)

    agent = Agent(parameters, optimizer_state)
    agent.training()

    parameters = load_state()
    visualize_agent()
