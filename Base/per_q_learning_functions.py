# py
from typing import Mapping, Tuple, Callable, List

# nn & rl
import gym
import jax
import optax
import haiku as hk
import jax.numpy as jnp
from jax.nn import one_hot
from numpy import ndarray, argmax, float64, reshape


def generate_per_train_step(optimizer: optax.adam, model: hk.Transformed) -> Callable:
    compute_per_loss: Callable = generate_per_loss_computation(model)

    @jax.jit
    def train_step(params: hk.Params,
                   opt_state: Mapping,
                   states: jnp.ndarray,
                   q_targets: jnp.ndarray,
                   is_weights: jnp.ndarray,
                   ) -> Tuple[hk.Params, Mapping]:
        grads = jax.grad(compute_per_loss)(params, states, q_targets, is_weights)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    return train_step


def generate_per_loss_computation(model: hk.Transformed) -> Callable:
    def compute_per_loss(params: hk.Params,
                         states: jnp.ndarray,
                         q_targets: jnp.ndarray,
                         is_weights: jnp.ndarray
                         ) -> jnp.ndarray:
        pred: jnp.ndarray = model.apply(params, states)
        loss_val: jnp.ndarray = jnp.mean(is_weights * jnp.sum(optax.huber_loss(pred, q_targets), axis=1), axis=0)
        return loss_val

    return compute_per_loss


def generate_priority_and_q_target_computation(model: hk.Transformed, gamma: float, env: gym.Env) -> Callable:
    @jax.jit
    def compute_priority_and_q_targets(params: hk.Params,
                                       target_params: hk.Params,
                                       states: jnp.ndarray,
                                       actions: ndarray,
                                       rewards: ndarray,
                                       observations: ndarray,
                                       dones: ndarray
                                       ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        q: ndarray = model.apply(params, states)
        next_q: ndarray = model.apply(params, observations)
        next_q_tm: ndarray = model.apply(target_params, observations)
        max_actions: ndarray = argmax(next_q, axis=1)
        targets: List = []
        for index, (max_action, action, done) in enumerate(zip(max_actions, actions, dones)):
            target_val: float = rewards[index] + (1.0-done) * (gamma*next_q_tm[index, max_action]-q[index, action])
            q_target: ndarray = q[index] + target_val * one_hot(action, env.action_space.n)
            targets.append(q_target)
        targets: jnp.ndarray = jnp.array(targets)
        two_dim_acs: ndarray = reshape(actions, (actions.shape[0], 1))
        error: jnp.ndarray = jnp.take_along_axis(targets, two_dim_acs, 1)-jnp.take_along_axis(q, two_dim_acs, 1)
        return error, targets

    return compute_priority_and_q_targets


@jax.jit
def per_preprocessing(states: ndarray,
                      actions: ndarray,
                      rewards: ndarray,
                      observations: ndarray,
                      dones: ndarray,
                      is_weights: ndarray
                      ) -> Tuple[jnp.ndarray, ndarray, ndarray, ndarray, ndarray, jnp.ndarray]:
    states: jnp.ndarray = jax.numpy.asarray(states)
    dones = dones.astype(float64)
    is_weights: jnp.ndarray = jax.numpy.asarray(is_weights)
    return states, actions, rewards, observations, dones, is_weights