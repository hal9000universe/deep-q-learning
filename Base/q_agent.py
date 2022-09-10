# py
import asyncio
from statistics import mean
from random import uniform
from typing import Mapping, List, Callable, Tuple, Dict
import matplotlib.pyplot as plt

# nn & rl
import gym
import jaxlib
import optax
import haiku as hk
import jax.numpy as jnp
from numpy import ndarray, average, asarray, float32
from numpy.random import randint
from sklearn.cluster import KMeans

# lib
from Base.replay_buffer import ReplayBuffer, sample_batch
from Base.q_learning_functions import action_computation, generate_q_target_comp, generate_train_step, preprocessing
from Base.utils import generate_saving, stop_time
from Base.metrics import generate_forward_analysis, loss_metric, data_means, mean_eq_dist


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
    _tpt_reward: float
    _reward_to_reach: float
    _num_actions: int
    # monitoring
    _loss_history: List[float]
    _episode_losses: List[ndarray]
    _reward_history: List[float]
    _episode_class_data: Dict[int, List[ndarray]]
    _class_mean_vars: List[ndarray]
    _episode_feature_data: Dict[int, List[ndarray]]
    _feature_mean_vars: List[ndarray]
    _tpt: bool
    # functions
    _compute_action: Callable
    _compute_q_targets: Callable
    _train_step: Callable
    _save_state: Callable
    _forward_analysis: Callable
    # ui
    _saving_directory: str
    _time_functions: bool
    _monitoring: bool
    _verbose: int

    def __init__(self,
                 network: hk.Transformed,
                 params: hk.Params,
                 optimizer: optax.adam,
                 opt_state: Mapping,
                 env: gym.Env,
                 buffer_size: int,
                 obs_shape: Tuple,
                 ac_shape: Tuple,
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
                 tpt_reward: float,
                 reward_to_reach: float,
                 num_actions: int,
                 saving_directory: str,
                 time_episodes: bool = False,
                 time_functions: bool = False,
                 monitoring: bool = False,
                 verbose: int = 1,
                 ):
        self._network = network
        self._params = params
        self._optimizer = optimizer
        self._opt_state = opt_state
        self._target_params = params
        self._env = env
        self._replay_buffer = ReplayBuffer(buffer_size=buffer_size,
                                           obs_shape=obs_shape,
                                           ac_shape=ac_shape)
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
        self._tpt_reward = tpt_reward
        self._reward_to_reach = reward_to_reach
        self._num_actions = num_actions
        self._reward_history = []
        self._compute_action = action_computation(network)
        self._compute_q_targets = generate_q_target_comp(network, gamma, env)
        self._train_step = generate_train_step(optimizer, network)
        self._save_state = generate_saving(saving_directory)
        self._saving_directory = saving_directory
        self._time_episodes = time_episodes
        self._time_functions = time_functions
        self._monitoring = monitoring
        self._verbose = verbose
        if self._monitoring:
            self._loss_history = []
            self._episode_losses = []
            self._episode_class_data = {i: [] for i in range(num_actions)}
            self._class_mean_vars = []
            self._episode_feature_data = {i: [] for i in range(num_actions)}
            self._feature_mean_vars = []
            self._tpt = False
            self._forward_analysis = generate_forward_analysis(network)

    async def _update_epsilon(self):
        self._epsilon = max(self._epsilon * self._epsilon_decay_rate, self._min_epsilon)

    async def _update_reward_history(self, episode_reward: float):
        self._reward_history.append(episode_reward)
        while len(self._reward_history) > 50:
            self._reward_history.pop(0)

    async def _update_loss_history(self):
        episode_loss: float = average(self._episode_losses)
        self._loss_history.append(episode_loss)
        while len(self._episode_losses) > 50:
            self._episode_losses.pop(0)

    def _average_reward(self) -> float:
        return mean(self._reward_history)

    def _policy(self, state: ndarray) -> int:
        if self._epsilon < uniform(0, 1):
            return int(self._compute_action(self._params, state))
        else:
            return randint(0, self._num_actions)

    async def _update_target_model(self):
        self._target_params = self._params

    def _tpt_analysis(self,
                      actions: jaxlib.xla_extension.DeviceArray,
                      activations: jaxlib.xla_extension.DeviceArray,
                      features: jaxlib.xla_extension.DeviceArray):
        actions: ndarray = asarray(actions, dtype=float32)
        activations: ndarray = asarray(activations, dtype=float32)
        features: ndarray = asarray(features, dtype=float32)
        for index, action in enumerate(actions):
            self._episode_class_data[action].append(activations[index])
        k_means = KMeans(self._num_actions)
        k_means.fit(features)
        labels = k_means.predict(features)
        for index, label in enumerate(labels):
            self._episode_feature_data[label].append(features[index])

    def _step(self):
        if self._time_functions:
            states, actions, rewards, observations, dones = stop_time("Sampling",
                                                                      sample_batch,
                                                                      self._replay_buffer.size,
                                                                      self._replay_buffer.states,
                                                                      self._replay_buffer.actions,
                                                                      self._replay_buffer.rewards,
                                                                      self._replay_buffer.observations,
                                                                      self._replay_buffer.dones,
                                                                      self._batch_size)
            states, actions, rewards, observations, dones = stop_time("Preprocessing",
                                                                      preprocessing,
                                                                      states,
                                                                      actions,
                                                                      rewards,
                                                                      observations,
                                                                      dones)
            q_targets: jnp.ndarray = stop_time("Q-target computation",
                                               self._compute_q_targets,
                                               self._params,
                                               self._target_params,
                                               states,
                                               actions,
                                               rewards,
                                               observations,
                                               dones)
            self._params, self._opt_state = stop_time("Train step",
                                                      self._train_step,
                                                      self._params,
                                                      self._opt_state,
                                                      states,
                                                      q_targets)
        else:
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
        if self._monitoring:
            activations, features = self._forward_analysis(self._params, states)
            loss: ndarray = loss_metric(activations, q_targets)
            self._episode_losses.append(loss)
            if self._tpt:
                self._tpt_analysis(actions, activations, features)

    def _reset(self):
        if self._monitoring and self._tpt:
            for index, cls in self._episode_class_data.items():
                self._episode_class_data[index] = cls[-101:-1]
            for index, ftr in self._episode_feature_data.items():
                self._episode_feature_data[index] = ftr[-101:-1]

    def _run_episode(self, step_count: int, episode: int):
        self._reset()
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
                self._step()

            if done:
                break

        if episode % self._replace_frequency == 0:
            asyncio.run(self._update_target_model())

        if episode % self._back_up_frequency == 0:
            self._save_state(self._params, self._opt_state)

        if self._monitoring and self._training_start < step_count:
            episode_loss: float = sum(self._episode_losses)
            self._loss_history.append(episode_loss)
            if self._tpt and step_count % self._train_frequency == 0:
                cls_means = data_means(self._episode_class_data)
                cls_eq_dist, cls_mean_var = mean_eq_dist(cls_means, 1)
                ftr_means = data_means(self._episode_feature_data)
                ftr_eq_dist, ftr_mean_var = mean_eq_dist(ftr_means, 0)
                self._class_mean_vars.append(cls_mean_var)
                self._feature_mean_vars.append(ftr_mean_var)

        asyncio.run(self._update_epsilon())
        asyncio.run(self._update_reward_history(epi_reward))
        if self._monitoring:
            asyncio.run(self._update_loss_history())

        return step_count

    def training(self):
        step_count: int = 0
        for episode in range(self._max_episodes):
            if self._time_episodes:
                step_count = stop_time("Time", self._run_episode, step_count, episode)
            else:
                step_count = self._run_episode(step_count, episode)

            if episode % 50 == 0 and self._verbose:
                print("Episode: {} -- Reward: {} -- Average: {}".format(episode,
                                                                        self._reward_history[-1],
                                                                        self._average_reward()))

            if self._average_reward() > self._tpt_reward:
                self._save_state(self._params, self._opt_state)
                self._tpt = True
                self._epsilon = 0.
                self._min_epsilon = 0.

            if self._average_reward() > self._reward_to_reach:
                self._save_state(self._params, self._opt_state)
                self._plot()
                return

    def evaluate(self) -> float:
        evaluation_runs: int = 10
        for run in range(evaluation_runs):
            state = self._env.reset()
            for step in range(self._max_steps):
                action = int(self._compute_action(self._params, state))
                state, reward, done, info = self._env.step(action)
        return self._average_reward()

    def _plot(self):
        loss_fig: plt.Figure = plt.figure()
        x_l = [i for i in range(len(self._loss_history))]
        plt.plot(x_l, self._loss_history)
        plt.xlabel("Episodes")
        plt.ylabel("Loss")
        loss_fig.show()
        # loss_fig.savefig(open(os.path.join("/", self._saving_directory, "/loss.png"), "wb"), dpi=250)
        if self._monitoring:
            reward_fig: plt.Figure = plt.figure()
            x_r = [i for i in range(len(self._reward_history))]
            plt.plot(x_r, self._reward_history)
            plt.xlabel("Episodes")
            plt.ylabel("Reward")
            reward_fig.show()
            # reward_fig.savefig(open(os.path.join("/", self._saving_directory, "/reward.png"), "wb"), dpi=250)
            x_v = [i for i in range(len(self._class_mean_vars))]
            cls_vars_fig: plt.Figure = plt.figure()
            plt.plot(x_v, self._class_mean_vars)
            plt.xlabel("Episodes")
            plt.ylabel("Class Mean Variance")
            cls_vars_fig.show()
            # cls_vars_fig.savefig(open(os.path.join(self._saving_directory, "/class_mean_variance.png"), "wb"), dpi=250)
            ftr_vars_fig: plt.Figure = plt.figure()
            plt.plot(x_v, self._feature_mean_vars)
            plt.xlabel("Episodes")
            plt.ylabel("Feature Mean Variance")
            ftr_vars_fig.show()
            # ftr_vars_fig.savefig(open(os.path.join(self._saving_directory, "/feature_mean_variance.png"), "wb"), dpi=250)
