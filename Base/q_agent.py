# py
import os
import time
import asyncio
from statistics import mean
from random import uniform
from pickle import load, dump

# nn & rl
from numpy.random import randint

# custom
from Base.replay_buffer import ReplayBuffer, sample_batch
from Base.q_learning_functions import *


class Agent:
    # rl
    _network: hk.Transformed
    _params: hk.Params
    _optimizer: optax.adam
    _opt_state: Mapping
    _env: gym.Env
    # hyperparameters
    _gamma: float
    _epsilon: float
    _epsilon_decay_rate: float
    _min_epsilon: float
    _max_episodes: int
    _max_steps: int
    _training_start: int
    _batch_size: int
    _train_frequency: int
    _back_up_frequency: int
    _replace_frequency: int
    # monitoring
    _episode_rewards: List[float]
    # functions
    _compute_action: Callable
    _compute_q_targets: Callable
    _train_step: Callable

    def __init__(self,
                 network: hk.Transformed,
                 params: hk.Params,
                 optimizer: optax.adam,
                 opt_state: Mapping,
                 env: gym.Env,
                 buffer_size: int,
                 obs_placeholder_shape: Tuple,
                 ac_placeholder_shape: Tuple,
                 gamma: float,
                 epsilon: float,
                 epsilon_decay_rate: float,
                 min_epsilon: float,
                 max_episodes: int,
                 max_steps: int,
                 training_start: int,
                 batch_size: int,
                 train_frequency: int,
                 back_up_frequency: int,
                 replace_frequency: int
                 ):
        self._network = network
        self._params = params
        self._optimizer = optimizer
        self._opt_state = opt_state
        self._target_params = params
        self._env = env
        self._replay_buffer = ReplayBuffer(buffer_size=buffer_size,
                                           obs_placeholder_shape=obs_placeholder_shape,
                                           ac_placeholder_shape=ac_placeholder_shape)
        self._gamma = gamma
        self._epsilon = epsilon
        self._epsilon_decay_rate = epsilon_decay_rate
        self._min_epsilon = min_epsilon
        self._max_episodes = max_episodes
        self._max_steps = max_steps
        self._training_start = training_start
        self._batch_size = batch_size
        self._train_frequency = train_frequency
        self._back_up_frequency = back_up_frequency
        self._replace_frequency = replace_frequency
        self._episode_rewards = []
        self._compute_action = action_computation(network)
        self._compute_q_targets = generate_q_target_computation(network, gamma, env)
        self._train_step = generate_train_step(optimizer, network)

    async def _update_epsilon(self):
        self._epsilon = max(self._epsilon * self._epsilon_decay_rate, self._min_epsilon)

    async def _update_episode_rewards(self, episode_reward: float):
        self._episode_rewards.append(episode_reward)
        while len(self._episode_rewards) > 50:
            self._episode_rewards.pop(0)

    def _average_reward(self) -> float:
        return mean(self._episode_rewards)

    def _policy(self, state: ndarray) -> int:
        if self._epsilon < uniform(0, 1):
            return int(self._compute_action(self._params, state))
        else:
            return randint(0, 4)

    async def _update_target_model(self):
        self._target_params = self._params

    def training(self):
        step_count: int = 0
        for episode in range(self._max_episodes):
            start: float = time.time()
            epi_reward: float = 0.
            state: ndarray = self._env.reset()
            for step in range(1, self._max_episodes + 1):
                step_count += 1
                action: int = self._policy(state)
                observation, reward, done, info = self._env.step(action)

                if step == self._max_steps:
                    done: bool = True

                self._replay_buffer.add(state[0], action, reward, observation[0], done)
                state = observation
                epi_reward += reward

                if self._replay_buffer.size >= self._training_start and step_count % self._train_frequency == 0:
                    states, actions, rewards, observations, dones = sample_batch(self._replay_buffer.size,
                                                                                 self._replay_buffer.states,
                                                                                 self._replay_buffer.actions,
                                                                                 self._replay_buffer.rewards,
                                                                                 self._replay_buffer.observations,
                                                                                 self._replay_buffer.dones,
                                                                                 self._batch_size)
                    states, actions, rewards, observations, dones = preprocessing(states,
                                                                                  actions,
                                                                                  rewards,
                                                                                  observations,
                                                                                  dones)
                    q_targets: jnp.ndarray = self._compute_q_targets(self._params,
                                                                     self._target_params,
                                                                     states,
                                                                     actions,
                                                                     rewards,
                                                                     observations,
                                                                     dones)
                    self._params, self._opt_state = self._train_step(self._params,
                                                                     self._opt_state,
                                                                     states,
                                                                     q_targets)

                if done:
                    break

            if episode % self._replace_frequency == 0:
                asyncio.run(self._update_target_model())

            if episode % self._back_up_frequency == 0:
                asyncio.run(save_training_state(self._params))

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
