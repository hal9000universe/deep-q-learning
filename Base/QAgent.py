from typing import List, Tuple, Optional
from statistics import mean
from random import uniform, randint

import tensorflow as tf
from tensorflow import Tensor, cast, convert_to_tensor, one_hot, GradientTape, function, Module
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.losses import Loss

import gym
from numpy import ndarray, argmax, float64

from Base.ReplayBuffer import ReplayBuffer


class DQN(Module):

    def __init__(self):
        super(DQN, self).__init__()

    def call(self, x: Tensor or ndarray, training: bool = False, mask: Optional[Tensor or ndarray] = None
             ) -> Tensor or ndarray:
        pass


class Agent:
    _replay_buffer: ReplayBuffer
    _q_model: DQN
    _target_model: DQN
    _model_version: int
    _epsilon: float
    _episode_rewards: List[float]
    _env: gym.Env
    _optimizer: Optimizer
    _loss: Loss
    _batch_size: int
    _max_steps: int
    _max_episodes: int
    _replace_frequency: int
    _back_up_frequency: int
    _training_start: int
    _train_frequency: int
    _epsilon_decay_rate: float
    _min_epsilon: float
    _gamma: float
    _manager: tf.train.CheckpointManager

    def __init__(self, buffer_size: int,
                 q_model: DQN,
                 epsilon: float,
                 env: gym.Env,
                 optimizer: Optimizer,
                 loss: Loss,
                 batch_size: int,
                 max_steps: int,
                 max_episodes: int,
                 replace_frequency: int,
                 back_up_frequency: int,
                 training_start: int,
                 train_frequency: int,
                 epsilon_decay_rate: float,
                 min_epsilon: float,
                 gamma: float,
                 manager: tf.train.CheckpointManager,
                 ):
        self._replay_buffer = ReplayBuffer(buffer_size=buffer_size, obs_shape=(buffer_size, 9))
        self._q_model = q_model
        self._target_model = DQN()
        self._target_model.set_weights(self._q_model.get_weights())
        self._model_version = 0
        self._epsilon = epsilon
        self._episode_rewards = []
        self._env = env
        self._optimizer = optimizer
        self._loss = loss
        self._batch_size = batch_size
        self._max_steps = max_steps
        self._max_episodes = max_episodes
        self._replace_frequency = replace_frequency
        self._back_up_frequency = back_up_frequency
        self._training_start = training_start
        self._train_frequency = train_frequency
        self._epsilon_decay_rate = epsilon_decay_rate
        self._min_epsilon = min_epsilon
        self._gamma = gamma
        self._manager = manager

    def _update_epsilon(self):
        self._epsilon = max(self._epsilon * self._epsilon_decay_rate, self._min_epsilon)

    def _update_episode_rewards(self, episode_reward: float):
        self._episode_rewards.append(episode_reward)
        while len(self._episode_rewards) > 50:
            self._episode_rewards.pop(0)

    def _average_reward(self) -> float:
        return mean(self._episode_rewards)

    def _policy(self, x: Tensor or ndarray) -> int or ndarray:
        if self._epsilon < uniform(0, 1):
            action: ndarray = argmax(self._q_model.advantage(x))
            return action
        else:
            action: int = randint(0, 3)
            return action

    def _update_target_model(self):
        self._target_model.set_weights(self._q_model.get_weights())

    def _compute_q_targets(self, states: Tensor, actions: ndarray, rewards: ndarray,
                           observations: Tensor, dones: ndarray) -> Tensor:
        q: Tensor = self._q_model(states)
        next_q: Tensor = self._q_model(observations)
        next_q_tm: Tensor = self._target_model(observations)
        max_actions: ndarray = argmax(next_q, axis=1)
        targets: List = []
        for index, action in enumerate(max_actions):
            if dones[index]:
                target_val: float = rewards[index]
            else:
                target_val: float = rewards[index] + self._gamma * next_q_tm[index, action] - q[index, actions[index]]
            q_target: Tensor = cast(q[index], dtype=float64) + one_hot(
                actions[index], self._env.action_space.n, on_value=cast(target_val, dtype=float64))
            targets.append(q_target)
        targets: Tensor = convert_to_tensor(targets, dtype=float64)
        return targets

    @function
    def _train_step(self, states: Tensor, q_targets: Tensor):
        with GradientTape() as tape:
            Q: Tensor = self._q_model(states)
            loss: Tensor = self._loss(q_targets, Q)
        grads: List[Tensor] = tape.gradient(loss, self._q_model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._q_model.trainable_variables))

    @staticmethod
    def _preprocess(batch: Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]
                    ) -> Tuple[Tensor, ndarray, ndarray, Tensor, ndarray]:
        states: Tensor = convert_to_tensor(batch[0], dtype=tf.float64)
        observations: Tensor = convert_to_tensor(batch[3], dtype=tf.float64)
        return states, batch[1], batch[2], observations, batch[4]

    def _build(self):
        test_input: ndarray = self._env.reset()
        self._q_model(test_input)
        self._target_model(test_input)

    def training(self):
        self._build()
        step_count: int = 0
        for episode in range(self._max_episodes):
            episode_reward: float = 0.
            state: ndarray = self._env.reset()
            for step in range(1, self._max_steps + 1):
                step_count += 1
                action: int = self._policy(state)
                observation, reward, done, info = self._env.step(action)

                if step == self._max_steps:
                    done = True

                self._replay_buffer.add(state[0], action, reward, observation[0], done)
                state = observation
                episode_reward += reward

                if self._replay_buffer.size >= self._training_start and step_count % self._train_frequency == 0:
                    batch: Tuple[ndarray, ndarray, ndarray, ndarray, ndarray] = self._replay_buffer.sample_batch(
                        self._batch_size
                    )
                    states, actions, rewards, observations, dones = self._preprocess(batch)
                    q_targets: Tensor = self._compute_q_targets(states, actions, rewards, observations, dones)
                    self._train_step(states, q_targets)

                if done:
                    break

                if step_count % self._replace_frequency == 0:
                    self._update_target_model()

            if episode % self._back_up_frequency == 0:
                self._manager.save()

            self._update_epsilon()
            self._update_episode_rewards(episode_reward)
            print("Episode: {} -- Reward: {} -- Average: {}".
                  format(episode, episode_reward, self._average_reward()))

    @property
    def model(self) -> DQN:
        return self._q_model

    @property
    def target_model(self) -> DQN:
        return self._target_model
