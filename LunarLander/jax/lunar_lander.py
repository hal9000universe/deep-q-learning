import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
import gym

import time
from typing import Callable, Mapping, Tuple, List, NamedTuple
from numpy import ndarray, zeros, float64, int64, argmax, append, newaxis
from jax.nn import one_hot
from numpy.random import uniform, randint

from statistics import mean


class TrainingState(NamedTuple):
    params: hk.Params
    avg_params: hk.Params
    opt_state: optax.OptState


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
        self._adv = hk.Linear(4)

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


def train_step(train_state: TrainingState, network: hk.Transformed, optimiser: optax.adam,
               loss_fn: Callable, states: jnp.ndarray, q_targets: jnp.ndarray):
    grads = jax.grad(loss_fn)(train_state.params, network, states, q_targets)
    updates, opt_state = optimiser.update(grads, train_state.opt_state)
    params = optax.apply_updates(train_state.params, updates)
    # Compute avg_params, the exponential moving average of the "live" params.
    # We use this only for evaluation (cf. https://doi.org/10.1137/0330046).
    avg_params = optax.incremental_update(
        params, train_state.avg_params, step_size=0.001)
    return TrainingState(params, avg_params, opt_state)


def compute_loss(params: hk.Params, network: hk.Transformed,
                 inp: jnp.ndarray, targ: jnp.ndarray) -> jnp.ndarray:
    pred: jnp.ndarray = network.apply(params, inp)
    loss_val: jnp.ndarray = jnp.sum(optax.l2_loss(pred, targ))
    return loss_val


class Agent:
    _replay_buffer: ReplayBuffer
    _q_model: hk.Transformed
    _model_version: int
    _epsilon: float
    _episode_rewards: List[float]

    def __init__(self, q_net, params: hk.Params, opt: optax.adam,
                 opt_state: Mapping, loss_fn: optax.huber_loss):
        buffer_size: int = 100000
        self._replay_buffer = ReplayBuffer(buffer_size=buffer_size, obs_shape=(buffer_size, 9))
        self._q_model = q_net
        self._train_state = TrainingState(params, params, opt_state)
        self._optimizer = opt
        self._loss = loss_fn
        self._params = params
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
            action: ndarray = argmax(self._q_model.apply(self._params, x))
            return int(action)
        else:
            return randint(0, 4)

    def _update_target_model(self):
        self._target_params = self._params

    def _compute_q_targets(self, states: ndarray, actions: ndarray, rewards: ndarray,
                           observations: ndarray, dones: ndarray) -> jnp.ndarray:
        q: ndarray = self._q_model.apply(self._params, states)
        next_q: ndarray = self._q_model.apply(self._params, observations)
        next_q_tm: ndarray = self._q_model.apply(self._target_params, observations)
        max_actions: ndarray = argmax(next_q, axis=1)
        targets: List = []
        for index, action in enumerate(max_actions):
            if dones[index]:
                target_val: float = rewards[index]
            else:
                target_val: float = rewards[index] + GAMMA * next_q_tm[index, action] - q[index, actions[index]]
            q_target: ndarray = q[index] + target_val * one_hot(actions[index], env.action_space.n)
            targets.append(q_target)
        targets: ndarray = jnp.array(targets)
        return targets

    def training(self):
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
                    states, actions, rewards, observations, dones = self._replay_buffer.sample_batch(BATCH_SIZE)
                    states: jnp.ndarray = jax.numpy.asarray(states)
                    q_targets: jnp.ndarray = self._compute_q_targets(states, actions, rewards, observations, dones)
                    self._train_state = train_step(self._train_state, self._q_model, self._optimizer, compute_loss, states, q_targets)
                    self._params = self._train_state.params

                if done:
                    break

            if episode % REPLACE_FREQUENCY == 0:
                self._update_target_model()

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
    BACKUP_FREQUENCY: int = 100
    TRAINING_START: int = 256
    TRAIN_FREQUENCY: int = 4
    EPSILON: float = 1.0
    EPSILON_DECAY_RATE: float = 0.995
    MIN_EPSILON: float = 0.001
    GAMMA: float = 0.999
    LEARNING_RATE: float = 0.001

    env: gym.Env = gym.make('LunarLander-v2')

    rng: jax.random.PRNGKeyArray = jax.random.PRNGKey(0)
    test_input: np.ndarray = zeros((1, 9))

    model: hk.Transformed = hk.without_apply_rng(hk.transform(lambda *args: Model()(*args)))
    optimizer: optax.adam = optax.adam(LEARNING_RATE)
    loss: Callable = optax.huber_loss

    parameters: hk.Params = model.init(rng, test_input)
    optimizer_state: Mapping = optimizer.init(parameters)

    agent = Agent(model, parameters, optimizer, optimizer_state, loss)
    agent.training()
