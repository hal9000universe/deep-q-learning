import jax
import jax.numpy as jnp
from jax import random, jit
from typing import Tuple, List
import time
import gym
from statistics import mean


class ReplayBuffer:
    _buffer_size: int
    _states: jnp.ndarray
    _actions: jnp.ndarray
    _rewards: jnp.ndarray
    _observations: jnp.ndarray
    _dones: jnp.ndarray
    _counter: int
    _samples: int
    _key: jax.xla.DeviceArray

    def __init__(self,
                 obs_shape: Tuple[int, int],
                 buffer_size: int = 1000000):
        self._buffer_size = buffer_size
        self._states = jnp.zeros(obs_shape, dtype=jnp.float64)
        self._actions = jnp.zeros((buffer_size,), dtype=jnp.int64)
        self._rewards = jnp.zeros((buffer_size,), dtype=jnp.float64)
        self._observations = jnp.zeros(obs_shape, dtype=jnp.float64)
        self._dones = jnp.zeros((buffer_size,), dtype=bool)
        self._counter = 0
        self._samples = 0
        self._key = random.PRNGKey(time.time_ns())

    @property
    def size(self) -> int:
        return self._samples

    def add(self, state: jnp.ndarray, action: int, reward: float, observation: jnp.ndarray, done: bool):
        self._states[self._counter % self._buffer_size] = state
        self._actions[self._counter % self._buffer_size] = action
        self._rewards[self._counter % self._buffer_size] = reward
        self._observations[self._counter % self._buffer_size] = observation
        self._dones[self._counter % self._buffer_size] = done
        self._counter += 1
        self._samples = min(self._counter, self._buffer_size)

    def sample_batch(self, batch_size: int = 64)\
            -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        self._key, subkey = random.split(self._key)
        random_indices: jnp.ndarray = random.randint(self._key, [batch_size], 0, self._samples - 1)
        batch = (self._states2[random_indices], self._actions[random_indices], self._rewards[random_indices],
                 self._observations[random_indices], self._dones[random_indices])
        return batch


class Agent:
    _replay_buffer: ReplayBuffer
    # _q_model: DDDQN
    # _target_model: DDDQN
    _model_version: int
    _epsilon: float
    _episode_rewards: List[float]
    _key = jax.xla.DeviceArray

    def __init__(self):
        buffer_size: int = 100000
        self._replay_buffer = ReplayBuffer(buffer_size=buffer_size, obs_shape=(buffer_size, 9))
        # self._q_model = q_net
        # self._target_model = DDDQN()
        # self._target_model.set_weights(self._q_model.get_weights())
        self._model_version = 0
        self._epsilon = EPSILON
        self._episode_rewards = []
        self._key = random.PRNGKey(time.time_ns())

    def _update_epsilon(self):
        self._epsilon = max(self._epsilon * EPSILON_DECAY_RATE, MIN_EPSILON)

    def _update_episode_rewards(self, episode_reward: float):
        self._episode_rewards.append(episode_reward)
        if len(self._episode_rewards) > 50:
            self._episode_rewards.pop(0)

    def _average_reward(self) -> float:
        return mean(self._episode_rewards)

    def _policy(self, x: jnp.ndarray) -> jnp.ndarray:
        self._key, subkey = random.split(self._key)
        if self._epsilon < random.uniform(self._key):
            # return argmax() TODO: output von netz berechnen
            pass
        else:
            self._key, subkey = random.split(self._key)
            return random.randint(self._key, (1,), 0, 4)

    def _update_target_model(self):
        pass

    def _compute_q_targets(self):
        pass

    def _train_step(self):
        pass

    def _build(self):
        test_input: jnp.ndarray = env.reset()
        test_input = jnp.append(test_input, 0.)
        test_input = test_input[jnp.newaxis, ...]
        # TODO: input in model

    def training(self):
        self._build()
        start: float = time.time()
        step_count: int = 0
        for episode in range(MAX_EPISODES):
            episode_reward: float = 0.
            state: jnp.ndarray = env.reset()
            state = jnp.append(state, 0.)
            state = state[jnp.newaxis, ...]
            for step in range(1, MAX_STEPS + 1):
                step_count += 1
                fraction_finished: float = step / MAX_STEPS
                action = self._policy(state)[0]
                observation, reward, done, info = env.step(action)
                observation = jnp.append(observation, fraction_finished)
                observation = observation[jnp.newaxis, ...]

                done = (step == MAX_STEPS)

                self._replay_buffer.add(state[0], action, reward, observation[0], done)
                state = observation
                episode_reward += reward

                if self._replay_buffer.size >= TRAINING_START and step_count & TRAIN_FREQUENCY == 0:
                    batch = self._replay_buffer.sample_batch(BATCH_SIZE)
                    # TODO: training

            if episode % REPLACE_FREQUENCY == 0:
                self._update_target_model()

            if episode % BACKUP_FREQUENCY == 0:
                # TODO: make save
                pass

            self._update_epsilon()
            self._update_episode_rewards(episode_reward)
            print("Episode: {} -- Reward: {} -- Average: {}".
                  format(episode, episode_reward, self._average_reward()))

        end: float = time.time()
        print('Time: {}s'.format(end - start))


if __name__ == "__main__":
    # hyperparameters
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
    REGULARIZATION_FACTOR: float = 0.001

    env: gym.Env = gym.make('LunarLander-v2')

