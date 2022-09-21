# py
import time
import asyncio
from statistics import mean
from random import uniform

# nn & rl
from numpy import vectorize
from numpy.random import randint, uniform

# lib
from Base.prioritized_experience_replay import PrioritizedExperienceReplay, sample_batch
from Base.per_q_learning_functions import *
from Base.q_learning_functions import action_computation
from Base.sum_tree import v_retrieve
from Base.utils import generate_saving


class PERAgent:
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
    _compute_priorities_and_q_targets: Callable
    _train_step: Callable
    _update_priorities: Callable
    _save_state: Callable

    def __init__(self,
                 network: hk.Transformed,
                 params: hk.Params,
                 optimizer: optax.adam,
                 opt_state: Mapping,
                 env: gym.Env,
                 buffer_size: int,
                 obs_placeholder_shape: Tuple,
                 ac_placeholder_shape: Tuple,
                 alpha: float,
                 beta: float,
                 min_priority: float,
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
                 replace_frequency: int,
                 reward_to_reach: float,
                 saving_directory: str,
                 ):
        self._network = network
        self._params = params
        self._optimizer = optimizer
        self._opt_state = opt_state
        self._target_params = params
        self._env = env
        self._per = PrioritizedExperienceReplay(buffer_size=buffer_size,
                                                obs_shape=obs_placeholder_shape,
                                                ac_shape=ac_placeholder_shape,
                                                alpha=alpha,
                                                beta=beta,
                                                min_priority=min_priority)
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
        self._reward_to_reach = reward_to_reach
        self._episode_rewards = []
        self._compute_action = action_computation(network)
        self._compute_priorities_and_q_targets = generate_priority_and_q_target_computation(network, gamma, env)
        self._train_step = generate_per_train_step(optimizer, network)
        self._update_priorities = vectorize(self._per.update)
        self._save_state = generate_saving(saving_directory)

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

                self._per.add_experience(0., state[0], action, reward, observation[0], done)
                state = observation
                epi_reward += reward

                if self._per.size >= self._training_start and step_count % self._train_frequency == 0:
                    indices: ndarray = v_retrieve(self._per.tree, self._batch_size)
                    states, actions, rewards, observations, dones, is_weights = sample_batch(self._per.size,
                                                                                             self._per.priorities,
                                                                                             self._per.states,
                                                                                             self._per.actions,
                                                                                             self._per.rewards,
                                                                                             self._per.observations,
                                                                                             self._per.dones,
                                                                                             indices,
                                                                                             self._batch_size,
                                                                                             self._per.alpha)
                    states, actions, rewards, observations, dones, is_weights = per_preprocessing(states,
                                                                                                  actions,
                                                                                                  rewards,
                                                                                                  observations,
                                                                                                  dones,
                                                                                                  is_weights)
                    priorities, q_targets = self._compute_priorities_and_q_targets(self._params,
                                                                                   self._target_params,
                                                                                   states,
                                                                                   actions,
                                                                                   rewards,
                                                                                   observations,
                                                                                   dones)
                    priorities = abs(priorities)
                    self._update_priorities(indices, priorities)
                    self._params, self._opt_state = self._train_step(self._params,
                                                                     self._opt_state,
                                                                     states,
                                                                     q_targets,
                                                                     is_weights)

                if done:
                    break

            if episode % self._replace_frequency == 0:
                asyncio.run(self._update_target_model())

            if episode % self._back_up_frequency == 0:
                self._save_state(self._params, self._opt_state)

            asyncio.run(self._update_epsilon())
            asyncio.run(self._update_episode_rewards(epi_reward))

            if self._average_reward() > self._reward_to_reach:
                self._save_state(self._params, self._opt_state)
                return

            end: float = time.time()
            if episode % 10 == 0:
                print("Episode: {} -- Reward: {} -- Average: {}".format(episode, epi_reward, self._average_reward()))
                print('Time: {}s'.format(end - start))
