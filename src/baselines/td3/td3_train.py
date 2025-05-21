# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Soft Actor-Critic training.

See: https://arxiv.org/pdf/1812.05905.pdf
"""

import functools
import time
from typing import Any, Callable, Optional, Tuple, Union, Generic, NamedTuple, Sequence

from absl import logging
from brax import base
from brax import envs
from brax.io import model
from brax.training import gradients
from brax.training import pmap
from brax.training.replay_buffers_test import jit_wrap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from src.baselines.td3 import td3_networks as td3_networks
from src.baselines.td3 import td3_losses as td3_losses
from brax.training.acme.types import NestedArray
from brax.training.types import Params, Policy
from brax.training.types import PRNGKey
from brax.v1 import envs as envs_v1
import flax
import jax
import jax.numpy as jnp
import optax
import copy
import os
import wandb

from envs.wrappers import TrajectoryIdWrapper
from src.evaluator import CrlEvaluator
from src.replay_buffer import QueueBase, Sample

from jax.scipy.stats import gaussian_kde

Metrics = types.Metrics
# Transition = types.Transition
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]


def soft_update(target_params: Params, online_params: Params, tau) -> Params:
    return jax.tree_util.tree_map(
        lambda x, y: (1 - tau) * x + tau * y, target_params, online_params
    )

class Transition(NamedTuple):
    """Container for a transition."""

    observation: NestedArray
    next_observation: NestedArray
    action: NestedArray
    reward: NestedArray
    discount: NestedArray
    extras: NestedArray = ()  # pytype: disable=annotation-type-mismatch  # jax-ndarray

sec_ver = False

def set_goal(
    env: Env,
    env_state: State,
    goals: NestedArray,
) -> State:
    pipeline_state = env_state.pipeline_state
    if sec_ver:
        updated_q = pipeline_state.q.at[:,-len(env.goal_indices_2):].set(goals) # Only set the position, not orientation
        updated_pipeline_state = pipeline_state.replace(q=updated_q)
        obs = jax.vmap(env._get_obs)(updated_pipeline_state)
        return env_state.replace(pipeline_state=updated_pipeline_state, obs=obs)#, obs=obs
    else:
        pipeline_state = env_state.pipeline_state
        q = pipeline_state.q.at[:,-len(env.goal_indices_2):].set(goals)
        qd = pipeline_state.qd.at[:,-len(env.goal_indices_2):].set(0)
        updated_pipeline_state = jax.vmap(env.pipeline_init)(q=q, qd=qd) 
        obs = jax.vmap(env._get_obs)(updated_pipeline_state)
        return env_state.replace(pipeline_state=updated_pipeline_state, obs=obs)
    
def set_goal_manipulation(
    env: Env,
    env_state: State,
    goals: NestedArray,
) -> State:
    return jax.vmap(env.update_goal)(env_state, goals)

def actor_step(
    env: Env,
    env_state: State,
    policy: Policy,
    goals,
    target_goals,
    goals_unachieved,
    target_unreached,
    key: PRNGKey,
    extra_fields: Sequence[str] = (),
    manipulation: bool = False,
    act_randomly: bool = False,
    continue_strategy: str = "uniform",
) -> Tuple[State, NestedArray, Transition]:
    """Collect data."""
    """if manipulation:
        env_state = set_goal_manipulation(env=env, env_state=env_state, goals=goals)
    else:
        env_state = set_goal(env=env, env_state=env_state, goals=goals)"""
    policy_actions, policy_extras = policy(env_state.obs, key)
    key, next_key = jax.random.split(key, 2)
    random_actions = jax.random.uniform(key, shape=policy_actions.shape, minval=-1, maxval=1)
    if continue_strategy == "uniform":
        actions = (1-jnp.array([goals_unachieved]).T) * random_actions + jnp.array([goals_unachieved]).T * policy_actions
    elif continue_strategy == "goal":
        actions = policy_actions
    elif continue_strategy == "target":
        actions = policy_actions
        goals = (1-jnp.array([goals_unachieved]).T) * target_goals + jnp.array([goals_unachieved]).T * goals
    #### !!!!! DEBUGGING !!!!!
    #exploration_noise = 0.4
    #noise_clip = 0.5
    #actions = (jax.random.normal(next_key, actions.shape) * exploration_noise).clip(-noise_clip, noise_clip)
    #### !!!!! DEBUGGING !!!!!

    #actions = int(act_randomly) * random_actions + (1-int(act_randomly)) * actions
    nstate = env.step(env_state, actions)
    dist_target = jnp.linalg.norm(env_state.obs[:,:len(env.goal_indices_2)] - jnp.array([target_goals[0]]), axis=1)
    target_unreached = target_unreached * jnp.array(dist_target > 0.1, dtype=float)
    # Update goals when reached
    goals_unachieved =  goals_unachieved * (1-nstate.reward)
    state_extras = {x: nstate.info[x] for x in extra_fields}
    return nstate, goals, target_goals, goals_unachieved, target_unreached, Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=env_state.obs,
        action=actions,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation=nstate.obs,
        extras={"policy_extras": policy_extras, "state_extras": state_extras},
    )


InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]

ReplayBufferState = Any

_PMAP_AXIS_NAME = "i"


class TrajectoryUniformSamplingQueue(QueueBase[Sample], Generic[Sample]):
    """Implements an uniform sampling limited-size replay queue BUT WITH TRAJECTORIES."""

    def sample_internal(self, buffer_state: ReplayBufferState) -> Tuple[ReplayBufferState, Sample]:
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f"Data shape expected by the replay buffer ({self._data_shape}) does "
                f"not match the shape of the buffer state ({buffer_state.data.shape})"
            )
        key, sample_key, shuffle_key = jax.random.split(buffer_state.key, 3)
        # NOTE: this is the number of envs to sample but it can be modified if there is OOM
        shape = self.num_envs

        # Sampling envs idxs
        envs_idxs = jax.random.choice(sample_key, jnp.arange(self.num_envs), shape=(shape,), replace=False)

        @functools.partial(jax.jit, static_argnames=("rows", "cols"))
        def create_matrix(rows, cols, min_val, max_val, rng_key):
            rng_key, subkey = jax.random.split(rng_key)
            start_values = jax.random.randint(subkey, shape=(rows,), minval=min_val, maxval=max_val)
            row_indices = jnp.arange(cols)
            matrix = start_values[:, jnp.newaxis] + row_indices
            return matrix

        @jax.jit
        def create_batch(arr_2d, indices):
            return jnp.take(arr_2d, indices, axis=0, mode="wrap")

        create_batch_vmaped = jax.vmap(create_batch, in_axes=(1, 0))

        matrix = create_matrix(
            shape,
            self.episode_length,
            buffer_state.sample_position,
            buffer_state.insert_position - self.episode_length,
            sample_key,
        )

        batch = create_batch_vmaped(buffer_state.data[:, envs_idxs, :], matrix)
        transitions = self._unflatten_fn(batch)
        return buffer_state.replace(key=key), transitions

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["config", "env"])
    def flatten_crl_fn(config, env, transition: Transition, sample_key: PRNGKey) -> Transition:
        if config.use_her:
            # Find truncation indexes if present
            seq_len = transition.observation.shape[0]
            arrangement = jnp.arange(seq_len)
            is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)
            single_trajectories = jnp.concatenate(
                [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len, axis=0
            )

            # final_step_mask.shape == (seq_len, seq_len)
            final_step_mask = is_future_mask * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5
            final_step_mask = jnp.logical_and(final_step_mask, transition.extras["state_extras"]["truncation"][None, :])
            non_zero_columns = jnp.nonzero(final_step_mask, size=seq_len)[1]

            key, sample_key = jax.random.split(sample_key)
            relable_strategy = jax.random.choice(key, jnp.array([0,1,2,3]), p=jnp.array([config.relabel_prob_future, config.relabel_prob_uniform, config.relabel_prob_geometric, 1-(config.relabel_prob_future + config.relabel_prob_uniform + config.relabel_prob_geometric)]), axis=0)

            # If final state is not present use original goal (i.e. don't change anything)
            # Uniform Future relabel strategy
            uniform_future_idx = ((seq_len-jnp.linspace(1,seq_len, seq_len)) * jax.random.uniform(key=sample_key, shape=(seq_len,)) + jnp.linspace(0,seq_len-1, seq_len)).astype(int)
            new_goals_uniform_future = transition.observation[uniform_future_idx][:, env.goal_indices]
            # Uniform relabel strategy
            uniform_idx = (seq_len * jax.random.uniform(key=sample_key, shape=(seq_len,))).astype(int)
            new_goals_uniform = transition.observation[uniform_idx][:, env.goal_indices]
            # Geomtric relabel strategy
            p = 0.2
            uniform = jax.random.uniform(sample_key, shape=(seq_len,))
            key, subkey = jax.random.split(sample_key)
            geom = jnp.ceil(jnp.log(1 - uniform) / jnp.log(1 - p))
            new_goals_idx = jnp.minimum(geom + jnp.linspace(0,seq_len-1, seq_len), jnp.ones(shape=(seq_len,))*(seq_len-1)).astype(int)
            new_goals_geometric = transition.observation[new_goals_idx][:, env.goal_indices]
            # Original goals
            new_goals_none = transition.observation[:, env.state_dim :]
            # choose randomly
            new_goals = (relable_strategy == 0).astype(int) * new_goals_uniform_future + (relable_strategy == 1).astype(int) * new_goals_uniform + (relable_strategy == 2).astype(int) * new_goals_geometric + (relable_strategy == 3).astype(int) * new_goals_none
            """
            if config.her_relabeling_strategy == "close_future":
                new_goals_idx = jnp.maximum(jnp.ones(shape=(seq_len,))*(seq_len-1), jax.random.randint(key=sample_key, minval=1,maxval=10,shape=(seq_len,))+jnp.linspace(0,seq_len-1, seq_len)).astype(int)
                new_goals = transition.observation[new_goals_idx][:, env.goal_indices]
            elif config.her_relabeling_strategy == "future": 
                uniform_future_idx = ((seq_len-jnp.linspace(1,seq_len, seq_len)) * jax.random.uniform(key=sample_key, shape=(seq_len,)) + jnp.linspace(0,seq_len-1, seq_len)).astype(int)
                key, subkey = jax.random.split(sample_key)
                relable = jax.random.choice(key, jnp.array([0,1]), p=jnp.array([1-config.relabel_prob, config.relabel_prob]), axis=0)
                new_goals = relable * transition.observation[uniform_future_idx][:, env.goal_indices] + (1-relable) *  transition.observation[:, env.state_dim :]
            elif config.her_relabeling_strategy == "geom_future":
                p = 0.1
                uniform = jax.random.uniform(sample_key, shape=(seq_len,))
                key, subkey = jax.random.split(sample_key)
                geom = jnp.ceil(jnp.log(1 - uniform) / jnp.log(1 - p))
                new_goals_idx = jnp.minimum(geom + jnp.linspace(0,seq_len-1, seq_len), jnp.ones(shape=(seq_len,))*(seq_len-1)).astype(int)
                relable = jax.random.choice(key, jnp.array([0,1]), p=jnp.array([1-config.relabel_prob, config.relabel_prob]), axis=0)
                new_goals = relable * transition.observation[new_goals_idx][:, env.goal_indices] + (1-relable) *  transition.observation[:, env.state_dim :]
            else:
                new_goals_idx = jnp.where(non_zero_columns == 0, arrangement, non_zero_columns)
                binary_mask = jnp.logical_and(non_zero_columns, non_zero_columns)
                new_goals = (
                    binary_mask[:, None] * transition.observation[new_goals_idx][:, env.goal_indices]
                    + jnp.logical_not(binary_mask)[:, None] * transition.observation[new_goals_idx][:, env.state_dim :]
                )
            """
            # Transform observation
            state = transition.observation[:, : env.state_dim]
            new_obs = jnp.concatenate([state, new_goals], axis=1)
            # Recalculate reward
            dist = jnp.linalg.norm(new_obs[:, env.state_dim :] - new_obs[:, env.goal_indices], axis=1)
            if config.negative_rewards:
                new_reward = jnp.array(dist < env.goal_reach_thresh, dtype=float)-1
            else:
                new_reward = jnp.array(dist < env.goal_reach_thresh, dtype=float)
            
            # Transform next observation
            next_state = transition.next_observation[:, : env.state_dim]
            new_next_obs = jnp.concatenate([next_state, new_goals], axis=1)

            return transition._replace(
                observation=jnp.squeeze(new_obs),
                next_observation=jnp.squeeze(new_next_obs),
                reward=jnp.squeeze(new_reward),
            )

        return transition


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    policy_optimizer_state: optax.OptState
    policy_params: Params
    target_policy_params: Params
    slow_target_policy_params: Params
    q_optimizer_state: optax.OptState
    q_params: Params
    target_q_params: Params
    metric_optimizer_state: optax.OptState
    metric_params: Params
    target_metric_params: Params
    gradient_steps: jnp.ndarray
    env_steps: jnp.ndarray
    normalizer_params: running_statistics.RunningStatisticsState
    achievement_params: Params
    achievement_optimizer_state: optax.OptState
    dynamics_params: Params
    dynamics_optimizer_state: optax.OptState
    alpha_optimizer_state: optax.OptState
    alpha_params: Params


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _init_training_state(
    key: PRNGKey,
    obs_size: int,
    local_devices_to_use: int,
    td3_network: td3_networks.Networks,
    policy_optimizer: optax.GradientTransformation,
    alpha_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation,
    metric_optimizer: optax.GradientTransformation,
    achievement_optimizer: optax.GradientTransformation,
    dynamics_optimizer: optax.GradientTransformation,
) -> TrainingState:
    """Inits the training state and replicates it over devices."""
    key_policy, key_q = jax.random.split(key)

    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
    alpha_optimizer_state = alpha_optimizer.init(log_alpha)

    policy_params = td3_network.policy_network.init(key_policy)
    policy_optimizer_state = policy_optimizer.init(policy_params)
    q_params = td3_network.q_network.init(key_q)
    q_optimizer_state = q_optimizer.init(q_params)
    metric_params = td3_network.metric_network.init(key_q)
    metric_optimizer_state = metric_optimizer.init(metric_params)
    achievement_params = td3_network.achievement_network.init(key_policy)
    achievement_optimizer_state = achievement_optimizer.init(achievement_params)
    dynamics_params = td3_network.dynamics_network.init(key_q)
    dynmamics_optimizer_state = dynamics_optimizer.init(dynamics_params)
    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
    alpha_optimizer_state = alpha_optimizer.init(log_alpha)

    normalizer_params = running_statistics.init_state(specs.Array((obs_size,), jnp.dtype("float32")))

    training_state = TrainingState(
        policy_optimizer_state=policy_optimizer_state,
        policy_params=policy_params,
        target_policy_params=policy_params,
        slow_target_policy_params=policy_params,
        q_optimizer_state=q_optimizer_state,
        q_params=q_params,
        target_q_params=q_params,
        metric_optimizer_state=metric_optimizer_state,
        metric_params=metric_params,
        target_metric_params=metric_params,
        gradient_steps=jnp.zeros(()),
        env_steps=jnp.zeros(()),
        normalizer_params=normalizer_params,
        achievement_params=achievement_params,
        achievement_optimizer_state=achievement_optimizer_state,
        dynamics_params=dynamics_params,
        dynamics_optimizer_state=dynmamics_optimizer_state,
        alpha_optimizer_state = alpha_optimizer_state,
        alpha_params = log_alpha
    )
    return jax.device_put_replicated(training_state, jax.local_devices()[:local_devices_to_use])


def train(
    environment: Union[envs_v1.Env, envs.Env],
    num_timesteps,
    episode_length: int,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_eval_envs: int = 128,
    learning_rate: float = 3e-4,
    discounting: float = 0.9,
    seed: int = 0,
    batch_size: int = 256,
    num_evals: int = 1,
    normalize_observations: bool = False,
    max_devices_per_host: Optional[int] = None,
    reward_scaling: float = 1.0,
    tau: float = 0.005,
    tau_policy: float = 0.005,
    min_replay_size: int = 0,
    max_replay_size: Optional[int] = None,
    deterministic_eval: bool = False,
    network_factory: types.NetworkFactory[td3_networks.Networks] = td3_networks.make_td3_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    train_step_multiplier: int = 1,
    unroll_length: int = 50,
    config: NamedTuple = None,
    checkpoint_logdir: Optional[str] = None,
    eval_env: Optional[envs.Env] = None,
    policy_delay: int = 2,
    noise_clip: int = 0.5,
    smoothing_noise: int = 0.2,
    exploration_noise: float = 0.4,
    randomization_fn: Optional[Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]] = None,
    visualization_interval: int = 5,
    goal_selection_strategy: str = "HER",
    ensemble_size: int = 2,
    ucb_mean_ini: int = 1.0,
    ucb_mean_tar: int = 1.0,
    ucb_std_ini:  int = 1.0,
    ucb_std_tar:  int = 1.0,
    filter_goals: str = "none",
    negative_rewards: bool = False,
    num_goals: int = 50000,
    mega_cutoff_step_size: int = 5,
    manipulation: bool = False,
    log_wandb: bool = False,
    goal_log_delay: int = 1,
    activate_final: bool = False,
    target_computation: bool = True,
    transform_ucb: bool = False,
    standardize_ucb: bool = False,
    critic_prior_file_name: str = None,
    obs_size_prior: int = None,
    action_size_prior: int = None,
    save_models: bool = False,
    ckpt_dir: str = "",
    use_metric: bool = False,
    eval_target_policy: bool = False,
    adaptation_strategy: str = "",
    continue_strategy: str = "uniform",
    grid_size: int = 30,
    adaptation_rate: int = 25,
    goal_achievement_target: float = 0.5,
    dir_var: float = 0.0,
    cur_alpha: float = 0.0,
    algo: str = "TD3",
):
    if save_models:
        checkpoint_logdir = ckpt_dir
    else:
        checkpoint_logdir = None
    negative_rewards = float(negative_rewards)
    mega_min_val = 0
    """TD3 training."""
    process_id = jax.process_index()
    local_devices_to_use = jax.local_device_count()
    if max_devices_per_host is not None:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    device_count = local_devices_to_use * jax.process_count()
    logging.info("local_device_count: %s; total_device_count: %s", local_devices_to_use, device_count)

    if min_replay_size >= num_timesteps:
        raise ValueError("No training will happen because min_replay_size >= num_timesteps")

    if max_replay_size is None:
        max_replay_size = num_timesteps

    # The number of environment steps executed for every `actor_step()` call.
    env_steps_per_actor_step = action_repeat * num_envs * unroll_length
    num_prefill_actor_steps = min_replay_size // unroll_length + 1
    print("Num_prefill_actor_steps: ", num_prefill_actor_steps)
    num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
    assert num_timesteps - min_replay_size >= 0
    num_evals_after_init = max(num_evals - 1, 1)
    # The number of epoch calls per training
    # equals to
    # ceil(num_timesteps - num_prefill_env_steps /
    #      (num_evals_after_init * env_steps_per_actor_step))
    num_training_steps_per_epoch = -(
        -(num_timesteps - num_prefill_env_steps) // (num_evals_after_init * env_steps_per_actor_step)
    )

    assert num_envs % device_count == 0
    env = environment
    if isinstance(env, envs.Env):
        jax.debug.print("standard")
        wrap_for_training = envs.training.wrap
    else:
        jax.debug.print("v1")
        wrap_for_training = envs_v1.wrappers.wrap_for_training

    rng = jax.random.PRNGKey(seed)
    rng, key = jax.random.split(rng)
    v_randomization_fn = None
    if randomization_fn is not None:
        v_randomization_fn = functools.partial(
            randomization_fn,
            rng=jax.random.split(key, num_envs // jax.process_count() // local_devices_to_use),
        )
    env = TrajectoryIdWrapper(env)
    env = wrap_for_training(
        env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )
    unwrapped_env = environment
    env_copy = copy.copy(env)
    obs_size = env.observation_size
    action_size = env.action_size

    normalize_fn = lambda x, y: x
    if normalize_observations:
        normalize_fn = running_statistics.normalize
    td3_network = network_factory(
        observation_size=obs_size, action_size=action_size, preprocess_observations_fn=normalize_fn, n_critics=ensemble_size, activate_final=activate_final, algo=algo
    )
    make_policy = td3_networks.make_inference_fn(td3_network, algo=algo)
    """
    Load critic prior
    """
    if critic_prior_file_name:
        critic_prior_file_name = os.path.expanduser(critic_prior_file_name)
        td3_network_prior = network_factory(
            observation_size=obs_size_prior, action_size=action_size_prior, preprocess_observations_fn=normalize_fn, n_critics=ensemble_size, activate_final=activate_final
        )
        critic_params_prior = model.load_params(critic_prior_file_name)[2]
        td3_network_prior.q_network.apply(None, critic_params_prior, jnp.zeros(shape=obs_size_prior), jnp.zeros(shape=action_size_prior))

    alpha_optimizer = optax.adam(learning_rate=3e-4)
    policy_optimizer = optax.adam(learning_rate=learning_rate)
    q_optimizer = optax.adam(learning_rate=learning_rate)
    metric_optimizer = optax.adam(learning_rate=learning_rate)
    achievement_optimizer = optax.adam(learning_rate=learning_rate)
    dynamics_optimizer = optax.adam(learning_rate=learning_rate)

    dummy_obs = jnp.zeros((obs_size,))
    dummy_action = jnp.zeros((action_size,))
    dummy_transition = Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=dummy_obs,
        next_observation=dummy_obs,
        action=dummy_action,
        reward=0.0,
        discount=0.0,
        extras={
            "state_extras": {
                "truncation": 0.0,
                "traj_id": 0.0,
            },
            "policy_extras": {},
        },
    )
    replay_buffer = jit_wrap(
        TrajectoryUniformSamplingQueue(
            max_replay_size=max_replay_size // device_count,
            dummy_data_sample=dummy_transition,
            sample_batch_size=batch_size // device_count,
            num_envs=num_envs,
            episode_length=episode_length,
        )
    )

    critic_loss, metric_loss, actor_loss, achievement_loss, dynamics_loss, alpha_loss = td3_losses.make_losses(
        td3_network=td3_network, reward_scaling=reward_scaling, discounting=discounting, smoothing=0.2, noise_clip=0.5, action_size=action_size, negative_rewards=negative_rewards, target_computation=target_computation, cur_alpha = cur_alpha, algo=algo
    )

    critic_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
        critic_loss, q_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
    )
    metric_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
        metric_loss, metric_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
    )
    actor_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
        actor_loss, policy_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
    )
    achievement_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
        achievement_loss, achievement_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
    )
    dynamics_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
        dynamics_loss, dynamics_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
    )
    alpha_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
        alpha_loss, alpha_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
    )

    def update_step(
        carry: Tuple[TrainingState, PRNGKey], transitions: Transition
    ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
        training_state, key = carry

        key, key_critic, key_metric, key_actor, key_dyn, key_alpha = jax.random.split(key, 6)
        critic_loss, q_params, q_optimizer_state = critic_update(
            training_state.q_params,
            training_state.target_q_params,
            training_state.target_policy_params,
            training_state.normalizer_params,
            training_state.dynamics_params,
            training_state.alpha_params,
            transitions,
            key_critic,
            optimizer_state=training_state.q_optimizer_state,
        )
        """metric_loss, metric_params, metric_optimizer_state = metric_update(
            training_state.metric_params,
            training_state.target_metric_params,
            training_state.target_policy_params,
            training_state.normalizer_params,
            training_state.alpha_params,
            transitions,
            key_metric,
            optimizer_state=training_state.metric_optimizer_state,
        )"""
        actor_loss, policy_params, policy_optimizer_state = actor_update(
            training_state.policy_params,
            training_state.q_params,
            training_state.normalizer_params,
            training_state.dynamics_params,
            training_state.alpha_params,
            transitions,
            key_actor,
            optimizer_state=training_state.policy_optimizer_state,
        )
        dynamics_loss, dynamics_params, dynamics_optimizer_state = dynamics_update(
            training_state.dynamics_params,
            training_state.normalizer_params,
            transitions,
            key_dyn,
            optimizer_state=training_state.dynamics_optimizer_state,
        )
        alpha_loss, alpha_params, alpha_optimizer_state = alpha_update(
            training_state.alpha_params,
            training_state.policy_params,
            training_state.normalizer_params,
            transitions,
            key_alpha,
            optimizer_state=training_state.alpha_optimizer_state,
        )

        def dont_policy_update(training_state):
            return (0.0, training_state.policy_params, training_state.policy_optimizer_state,
                    training_state.target_q_params, training_state.target_policy_params, training_state.slow_target_policy_params)

        def do_policy_update(training_state):
            actor_loss, policy_params, policy_optimizer_state = actor_update(
                training_state.policy_params,
                training_state.q_params,
                training_state.normalizer_params,
                training_state.dynamics_params,
                training_state.alpha_params,
                transitions,
                key_actor,
                optimizer_state=training_state.policy_optimizer_state,
            )
            new_target_q_params = soft_update(training_state.target_q_params, q_params, tau)
            #new_target_metric_params = soft_update(training_state.target_metric_params, metric_params, tau)
            new_target_policy_params = soft_update(training_state.policy_params, policy_params, tau)
            new_slow_target_policy_params = soft_update(training_state.slow_target_policy_params, policy_params, tau_policy)
            return (actor_loss, policy_params, policy_optimizer_state,
                    new_target_q_params, new_target_policy_params, new_slow_target_policy_params)

        update_policy = training_state.gradient_steps % policy_delay == 0
        (actor_loss, policy_params,
         policy_optimizer_state, new_target_q_params, new_target_policy_params, new_slow_target_policy_params)\
            = jax.lax.cond(update_policy, do_policy_update, dont_policy_update, training_state)

        metrics = {
            "critic_loss": critic_loss,
            #"metric_loss": metric_loss,
            "actor_loss": actor_loss,
            "dynamics_loss": dynamics_loss,
        }

        new_training_state = TrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            target_policy_params=new_target_policy_params,
            slow_target_policy_params=new_slow_target_policy_params,
            q_optimizer_state=q_optimizer_state,
            q_params=q_params,
            target_q_params=new_target_q_params,
            metric_optimizer_state=training_state.metric_optimizer_state,
            metric_params=training_state.metric_params,
            target_metric_params=training_state.target_metric_params,
            gradient_steps=training_state.gradient_steps + 1,
            env_steps=training_state.env_steps,
            normalizer_params=training_state.normalizer_params,
            achievement_params=training_state.achievement_params,
            achievement_optimizer_state=training_state.achievement_optimizer_state,
            dynamics_params = dynamics_params,
            dynamics_optimizer_state = dynamics_optimizer_state,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params,
        )
        return (new_training_state, key), metrics

    def get_initial_training_state(
        goals: NestedArray,
        key: PRNGKey,
    ) -> State:
        env_keys = jax.random.split(key, goals.shape[0])
        env_keys = jnp.reshape(env_keys, (local_devices_to_use, -1) + env_keys.shape[1:])
        initial_state = jax.pmap(env_copy.reset)(env_keys)
        q = initial_state.pipeline_state.q[0].at[:,-len(env.goal_indices_2):].set(goals)
        qd = initial_state.pipeline_state.qd[0].at[:,-len(env.goal_indices_2):].set(0)
        initial_state = jax.vmap(env.pipeline_init)(q=q, qd=qd)
        return initial_state

    def compute_value_function(
        training_state: TrainingState,
        observations: NestedArray,
        transform_ucb: bool = False,
        use_prior: bool = False,
        use_metric: bool = False
    ) -> Tuple[NestedArray, NestedArray]:
        if algo in ["TD3", "ThompTD3"]:
            if use_prior:
                actions = jnp.zeros(shape=(observations.shape[0], action_size_prior))
                observations = jnp.concat([observations[:,:2], jnp.zeros(shape=(observations.shape[0], 2)), observations[:,-2:]], axis=1)#jnp.zeros(shape=(observations.shape[0], 2)), 
                current_q = td3_network_prior.q_network.apply(None, critic_params_prior, observations, actions)
            else:
                actions = td3_network.policy_network.apply(training_state.normalizer_params, training_state.target_policy_params, observations)
                standard_q = td3_network.q_network.apply(training_state.normalizer_params, training_state.q_params, observations, actions)
                metric_q = td3_network.metric_network.apply(training_state.normalizer_params, training_state.metric_params, observations, actions)
                if use_metric:
                    current_q = metric_q
                else:
                    current_q = standard_q
        else:
            dist_params = td3_network.policy_network.apply(training_state.normalizer_params, training_state.policy_params, observations)
            action = td3_network.parametric_action_distribution.sample_no_postprocessing(dist_params, key)
            #log_prob = td3_network.parametric_action_distribution.log_prob(dist_params, action)
            action = td3_network.parametric_action_distribution.postprocess(action)
            current_q = td3_network.q_network.apply(training_state.normalizer_params,training_state.target_q_params,observations,action)
        if transform_ucb and not negative_rewards:
            current_q = jnp.maximum(current_q+jnp.ones(shape=current_q.shape)*0.01, jnp.ones(shape=current_q.shape)*0.001)
            current_q = -jnp.log((1-discounting)*current_q)/jnp.log(discounting) # neccessary due to log_\gamma instead of log_e
        else:
            transformed_q = jnp.maximum(current_q+jnp.ones(shape=current_q.shape)*0.01, jnp.ones(shape=current_q.shape)*0.001)
            transformed_q = -jnp.log((1-discounting)*transformed_q)/jnp.log(discounting) # neccessary due to log_\gamma instead of log_e
            return jnp.mean(current_q, axis=1), jnp.nan_to_num(jnp.std(transformed_q, axis=1), nan=0.0)
        return jnp.mean(current_q, axis=1), jnp.nan_to_num(jnp.std(current_q, axis=1), nan=0.0)
    
    def get_observations(
        env_state: Union[envs.State, envs_v1.State],
        goals: NestedArray,
        ):
        if manipulation:
            observations_initial = jax.vmap(env._get_obs)(env_state, goals, jnp.zeros(goals.shape))
        else:
            observations_initial = jax.vmap(env._get_obs)(env_state)
        return observations_initial

    def standardize(
        array: NestedArray
    ) -> NestedArray:
        return (array-jnp.min(array))/jnp.max(array-jnp.min(array)+0.01)
    
    def goal_density_estimator(
        goals_kde: NestedArray
    ) -> Tuple[gaussian_kde, NestedArray, NestedArray]:
        mean = jnp.mean(goals_kde, axis=0)
        std = jnp.std(goals_kde, axis=0)
        dataset = (goals_kde-mean)/std
        kde = gaussian_kde(dataset=dataset.T) # includes length_scale fitting
        return kde, mean, std

    # If enough goals achievable will lead to goals_selection being applied immediately
    # Else choose remaining goals that have leads achievable scaling (are the most achievable among the rest of the goals)
    def filter_achieved_goals(
            goals_selection: NestedArray,
            filter_goals: str,
            achievable: NestedArray,
            achievable_scaling: NestedArray, # Lower -> less achievable
            out_type: str, # MAX, MIN, PROBS
        ) -> NestedArray:
        achievable_scaling = achievable_scaling - jnp.max(achievable_scaling)
        if filter_goals == "MEGA": 
            if out_type == "PROBS":
                values = achievable * goals_selection - (1-achievable) * (100 * (jnp.max(jnp.abs(goals_selection)) - achievable_scaling))
                values = values - jnp.min(values)
                probs = values / jnp.sum(values)
                #probs = jnp.exp(values)/jnp.sum(jnp.exp(values))
                return probs
            elif out_type == "MAX":
                return achievable * goals_selection - (1-achievable) * (2 * jnp.max(jnp.abs(goals_selection)) - achievable_scaling)
            elif out_type == "MIN":
                return achievable * goals_selection + (1-achievable) * (2 * jnp.max(jnp.abs(goals_selection)) - achievable_scaling)
            else:
                raise ValueError
        else:
            if out_type == "PROBS":
                goals_selection = goals_selection - jnp.min(goals_selection)
                return goals_selection / jnp.sum(goals_selection)
                #return jnp.exp(goals_selection)/jnp.sum(jnp.exp(goals_selection))
            else:
                return goals_selection

    def goal_selection(
        training_state: TrainingState,
        strategy: str,
        buffer_state: ReplayBufferState,
        env_state: Union[envs.State, envs_v1.State],
        key: PRNGKey,
        mega_cutoff: int,
        ucb_params: jnp.array,
    ) -> Tuple[NestedArray, ReplayBufferState]:
        key, next_key = jax.random.split(key)
        target_goals = env_state.obs[:,-len(env.goal_indices_2):]
        observation_list = []
        for _ in range(int((2*num_goals)/(batch_size*unroll_length))+1):
            buffer_state, sample = replay_buffer.sample_internal(buffer_state)
            observation_list.append(sample.observation)
        observations = jnp.concatenate(observation_list)
        observations = jax.random.permutation(key, jnp.squeeze(jnp.concatenate(observations, axis=0))) #sample. Does this yield the correct samples?
        key, next_key = jax.random.split(next_key)
        kde, mean, std = goal_density_estimator(observations[-num_goals:,env.goal_indices_2])
        min_distance_target_all = jnp.min(jnp.linalg.norm(observations[:,env.goal_indices_2]-target_goals[0], axis=1)) # For tracking / debugging
        observations = observations[:num_goals]
        achieved_goals = observations[:,env.goal_indices_2]
        max_distance_goal = jnp.max(jnp.linalg.norm(achieved_goals, axis=1))    # Sanity check
        log_likelihood = kde.logpdf(((achieved_goals-mean)/std).T)
        key, next_key = jax.random.split(next_key)
        initial_state = get_initial_training_state(achieved_goals, key)
        observations_initial = get_observations(initial_state, achieved_goals)
        mean_ini_goal, std_ini_goal = compute_value_function(training_state, observations_initial)
        achievable_mega = (mean_ini_goal > jnp.mean(jnp.array(mega_cutoff))).astype(float)
        achievable_scaling = mean_ini_goal
        correlation = 0
        random_states = achieved_goals[:num_envs]
        unif_goals = jnp.linspace(-env.max_dis, env.max_dis, grid_size)
        unif_goals_grid_1, unif_goals_grid_2 = jnp.meshgrid(unif_goals, unif_goals)
        unif_goals_grid = jnp.concatenate([jnp.reshape(unif_goals_grid_1, (-1,1)), jnp.reshape(unif_goals_grid_2, (-1,1))], axis=1)
        grid_bin = jnp.zeros(unif_goals_grid.shape)
        if grid_size > 0:
            diff = unif_goals_grid[:, jnp.newaxis, :] - achieved_goals[jnp.newaxis, :, :]
            norms = jnp.linalg.norm(diff, axis=2)
            indices_goals = jnp.argmin(norms, axis=0)
            grid_bin = grid_bin.at[indices_goals].set(jnp.ones((indices_goals.shape[0],2)))
        if strategy == "HER":
            return target_goals, target_goals, buffer_state, 0, 0, 0, max_distance_goal, random_states, grid_bin
        elif strategy == "RANDOM":
            goals = achieved_goals[:num_envs]
        elif strategy == "UNIFORM":
            probs = filter_achieved_goals(goals_selection=-log_likelihood, filter_goals=filter_goals, achievable=achievable_mega, achievable_scaling=achievable_scaling, out_type="PROBS")
            indices = jax.random.choice(key, a = jnp.linspace(0,num_goals-1,num_goals).astype(int), p = probs, shape=(num_envs,), replace=False)
            goals = achieved_goals[indices]
        elif strategy == "MEGA":
            log_likelihood = filter_achieved_goals(goals_selection=log_likelihood, filter_goals=filter_goals, achievable=achievable_mega, achievable_scaling=achievable_scaling, out_type="MIN")
            indices = jnp.argpartition(log_likelihood, num_envs)[:num_envs]
            goals = achieved_goals[indices]
        elif strategy == "UCB" or strategy ==  "ORACLE":
            final_objective = target_goals[0]
            if dir_var > 0:
                final_objective += jax.random.normal(key, final_objective.shape) * dir_var
            oracle_distance = -jnp.linalg.norm(achieved_goals-final_objective, axis=1) * (unroll_length/env.max_dis)*2
            mean_ini_goal, std_ini_goal = compute_value_function(training_state, observations_initial, transform_ucb=transform_ucb)
            observations = jax.vmap(lambda x: x.at[-len(env.goal_indices_2):].set(target_goals[0]))(observations)  # Set observation goals to target goals
            mean_goal_tar, std_goal_tar = compute_value_function(training_state, observations, transform_ucb=transform_ucb, use_metric=use_metric)
            if strategy == "ORACLE":
                mean_goal_tar = oracle_distance
            elif critic_prior_file_name:
                mean_goal_tar, std_goal_tar = compute_value_function(training_state, observations, transform_ucb=transform_ucb, use_prior=True)
            if standardize_ucb:
                mean_ini_goal = standardize(mean_ini_goal) 
                std_ini_goal = standardize(std_ini_goal) 
                mean_goal_tar = standardize(mean_goal_tar) 
                std_goal_tar = standardize(std_goal_tar)
            correlation = jnp.corrcoef(mean_goal_tar, oracle_distance)[0][1]
            correlation = jnp.nan_to_num(correlation, nan=0)
            ucb = ucb_params[0] * mean_ini_goal + ucb_params[1] * mean_goal_tar + ucb_params[2] * std_ini_goal + ucb_params[3] * std_goal_tar
            ucb = filter_achieved_goals(goals_selection=ucb, filter_goals=filter_goals, achievable=achievable_mega, achievable_scaling=achievable_scaling, out_type="MAX")
            indices = jnp.argpartition(ucb, -num_envs)[-num_envs:]
            goals = achieved_goals[indices]
        else:
            raise NotImplementedError
        min_distance_target_goals = jnp.min(jnp.linalg.norm(goals-target_goals[0], axis=1)) # For tracking / debugging
        return goals, target_goals, buffer_state, min_distance_target_all, min_distance_target_goals, correlation, max_distance_goal, random_states, grid_bin
    
    # MEGA cutoff strategy: Adjust cutoff parameter if too many or too little goals were achieved
    def adapt_mega_params(mega_cutoff, mean_ini_goal, goals_achieved):
        furthest_achieved = jnp.min(mean_ini_goal * goals_achieved)
        mega_cutoff_mean = jnp.mean(jnp.array(mega_cutoff))
        if negative_rewards:
            cutoff_decrease = jax.lax.cond(jnp.logical_and(jnp.mean(goals_achieved) > 0.7, jnp.logical_and(mega_cutoff_mean > furthest_achieved, mega_cutoff_mean > -0.8*episode_length)), lambda _: 1, lambda _: 0, operand=None)
            cutoff_increase = jax.lax.cond(jnp.logical_and(jnp.mean(goals_achieved) < 0.3, mega_cutoff_mean+mega_cutoff_step_size <= -3), lambda _: 1, lambda _: 0, operand=None)
        else:
            cutoff_decrease = jax.lax.cond(jnp.logical_and(jnp.mean(goals_achieved) > 0.7, jnp.logical_and(mega_cutoff_mean > furthest_achieved, mega_cutoff_mean-mega_cutoff_step_size >= mega_min_val)), lambda _: 1, lambda _: 0, operand=None)
            cutoff_increase = jax.lax.cond(jnp.logical_and(jnp.mean(goals_achieved) < 0.3,  mega_cutoff_mean < 0.8*episode_length), lambda _: 1, lambda _: 0, operand=None)
        mega_cutoff += (cutoff_increase-cutoff_decrease).astype(int) * mega_cutoff_step_size
        return mega_cutoff

    def adapt_ucb_params(training_state, goal_achieval_rate, ucb_params):
        ucb_mean_ini_cur, ucb_mean_tar_cur, ucb_std_ini_cur, ucb_std_tar_cur = jnp.mean(ucb_params[0]), jnp.mean(ucb_params[1]), jnp.mean(ucb_params[2]), jnp.mean(ucb_params[3])
        if adaptation_strategy == "sch":
            ucb_mean_tar_cur += 2/(num_timesteps/episode_length)
            return jnp.array([ucb_mean_ini_cur, ucb_mean_tar_cur, ucb_std_ini_cur, ucb_std_tar_cur])
        elif adaptation_strategy == "opt":
            return jnp.array([ucb_mean_ini_cur, ucb_mean_tar_cur, ucb_std_ini_cur, ucb_std_tar_cur])
        elif adaptation_strategy in ["simple", "simple_sch"]:
            diff = 0.2
            if ucb_mean_ini == 0 and ucb_mean_tar == 0:
                mid_point_ens_exp = (ucb_mean_ini+ucb_std_ini)/2
                step_size_ens_exp = jnp.mean(mid_point_ens_exp)/adaptation_rate
                ucb_mean_ini_cur -= (jnp.mean(goal_achieval_rate) > goal_achievement_target+diff).astype(int) * step_size_ens_exp 
                ucb_mean_ini_cur += (jnp.mean(goal_achieval_rate) < goal_achievement_target-diff).astype(int) * step_size_ens_exp
                ucb_mean_ini_cur = jnp.mean(jnp.maximum(jnp.minimum(ucb_mean_ini_cur, 2*mid_point_ens_exp), 0))
            else:
                mid_point = (ucb_mean_ini+ucb_mean_tar)/2
                step_size = mid_point / adaptation_rate
                ucb_mean_tar_cur += (jnp.mean(goal_achieval_rate) > goal_achievement_target+diff).astype(int) * step_size 
                ucb_mean_tar_cur -= (jnp.mean(goal_achieval_rate) < goal_achievement_target-diff).astype(int) * step_size
                ucb_mean_tar_cur = jnp.maximum(jnp.minimum(ucb_mean_tar_cur, 2*mid_point), 0)
                ucb_mean_ini_cur = 2*mid_point-ucb_mean_tar_cur
            return jnp.array([ucb_mean_ini_cur, ucb_mean_tar_cur, ucb_std_ini_cur, ucb_std_tar_cur])
        else:
            return jnp.array([ucb_mean_ini_cur, ucb_mean_tar_cur, ucb_std_ini_cur, ucb_std_tar_cur])

    def get_experience(
        training_state: TrainingState,
        normalizer_params: running_statistics.RunningStatisticsState,
        policy_params: Params,
        env_state: Union[envs.State, envs_v1.State],
        buffer_state: ReplayBufferState,
        strategy: str,
        key: PRNGKey,
        mega_cutoff: int,
        act_randomly: bool,
        ucb_params: jnp.array,
    ) -> Tuple[
        running_statistics.RunningStatisticsState,
        Union[envs.State, envs_v1.State],
        ReplayBufferState,
    ]:
        key, next_key = jax.random.split(key)
        policy = make_policy((normalizer_params, policy_params), exploration_noise=exploration_noise, noise_clip=noise_clip)
        key, next_key = jax.random.split(key)
        goals, target_goals, buffer_state, min_distance_target_all, min_distance_target_goals, correlation, max_distance_goal, random_states, grid_bin = goal_selection(training_state=training_state, buffer_state=buffer_state, env_state=env_state, key=next_key, strategy=strategy, mega_cutoff=mega_cutoff, ucb_params=ucb_params)
        if manipulation:
            env_state = set_goal_manipulation(env=env, env_state=env_state, goals=goals)
        else:
            env_state = set_goal(env=env, env_state=env_state, goals=goals)
        goals_unachieved = jnp.ones(num_envs) # 1: unachieved, 0: achieved
        target_unreached = jnp.ones(num_envs) # 1: unachieved, 0: achieved
        @jax.jit
        def f(carry, unused_t):
            env_state, goals, target_goals, goals_unachieved, target_unreached, current_key = carry
            
            current_key, next_key = jax.random.split(current_key)
            env_state, goals, target_goals, goals_unachieved, target_unreached, transition = actor_step(
                env,
                env_state,
                policy,
                goals,
                target_goals,
                goals_unachieved,
                target_unreached,
                current_key,
                act_randomly=True, # For testing 
                extra_fields=(
                    "truncation",
                    "traj_id",
                ),
                manipulation=manipulation,
                continue_strategy = continue_strategy,
            )
            return (env_state, goals, target_goals, goals_unachieved, target_unreached, next_key), transition

        (env_state, _, target_goals, goals_unachieved, target_unreached, _), data = jax.lax.scan(f, (env_state, goals, target_goals, goals_unachieved, target_unreached, key), (), length=unroll_length)
        goals_achieved = 1-goals_unachieved
        key, next_key = jax.random.split(next_key)
        initial_state = get_initial_training_state(goals, key)
        observations_initial = get_observations(initial_state, goals)
        mean_ini_goal, std_ini_goal = compute_value_function(training_state, observations_initial)
        observations_target = jax.vmap(lambda x,y: x.at[env.goal_indices].set(y))(observations_initial, goals)
        observations_target = jax.vmap(lambda x: x.at[-len(env.goal_indices_2):].set(target_goals[0]))(observations_target)  # Set observation goals to target goals
        mean_goal_tar, std_goal_tar = compute_value_function(training_state, observations_target)
        # MEGA cutoff strategy: Adjust cutoff parameter if too many or too little goals were achieved
        mega_cutoff = adapt_mega_params(mega_cutoff, mean_ini_goal, goals_achieved)
        #ucb_mean_ini, ucb_mean_tar, ucb_std_ini, ucb_std_tar = adapt_ucb_params()
        normalizer_params = running_statistics.update(
            normalizer_params,
            jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
            ).observation,  # so that batch size*unroll_length is the first dimension
            pmap_axis_name=_PMAP_AXIS_NAME,
        )
        buffer_state = replay_buffer.insert(buffer_state, data)
        mean_ini_goal_mean = jnp.mean(mean_ini_goal)
        goals = jnp.concatenate([goals, jnp.array([mean_ini_goal]).T, jnp.array([std_ini_goal]).T, jnp.array([mean_goal_tar]).T, jnp.array([std_goal_tar]).T, jnp.array([goals_unachieved]).T, random_states, jnp.array([target_unreached]).T], axis=1)

        ### Create Table of value function of grid of goals (only ant maze)
        grid_table = jnp.zeros(shape=(1**env.goal_indices.shape[0], env.goal_indices.shape[0]+4))
        if grid_size > 0:
            unif_goals = jnp.linspace(-env.max_dis, env.max_dis, grid_size)
            unif_goals_grid_1, unif_goals_grid_2 = jnp.meshgrid(unif_goals, unif_goals)
            unif_goals_grid = jnp.concatenate([jnp.reshape(unif_goals_grid_1, (-1,1)), jnp.reshape(unif_goals_grid_2, (-1,1))], axis=1)
            if len(env.goal_indices) > 2:
                final_grid = []
                for h in env.grid_heights:
                    final_grid.append(jnp.concatenate([unif_goals_grid, jnp.ones((unif_goals_grid.shape[0],1))*h], axis=1))
                unif_goals_grid = jnp.vstack(final_grid)
                #jax.debug.print("{}", unif_goals_grid.shape)
            state_ini_goal = get_initial_training_state(unif_goals_grid, key)
            obs_ini_goal = get_observations(state_ini_goal, unif_goals_grid)
            grid_mean_ini_goal, grid_std_ini_goal = compute_value_function(training_state, obs_ini_goal, transform_ucb=transform_ucb)
            obs_goal_target = jax.vmap(lambda x,y: x.at[env.goal_indices].set(y))(obs_ini_goal, unif_goals_grid)
            obs_goal_target = jax.vmap(lambda x: x.at[-len(env.goal_indices_2):].set(target_goals[0]))(obs_goal_target)  # Set observation goals to target goals
            grid_mean_goal_tar, grid_std_goal_tar = compute_value_function(training_state, obs_goal_target, transform_ucb=transform_ucb)
            grid_table = jnp.concatenate([unif_goals_grid, jnp.reshape(grid_mean_ini_goal, (-1,1)), jnp.reshape(grid_std_ini_goal, (-1,1)), jnp.reshape(grid_mean_goal_tar, (-1,1)), jnp.reshape(grid_std_goal_tar, (-1,1)), grid_bin], axis=1)
        else: 
            grid_table = 0
        
        return normalizer_params, env_state, buffer_state, mega_cutoff, jnp.mean(goals_achieved), jnp.squeeze(mean_ini_goal_mean), goals, min_distance_target_all, min_distance_target_goals, correlation, max_distance_goal, grid_table, jnp.mean(target_unreached)

    def training_step(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
        mega_cutoff: int,
        ucb_params: jnp.array,
    ) -> Tuple[TrainingState, Union[envs.State, envs_v1.State], ReplayBufferState, Metrics, int]:
        experience_key, training_key = jax.random.split(key)
        normalizer_params, env_state, buffer_state, mega_cutoff, goal_achieval_rate, mean_ini_goal_mean, goals, min_distance_target_all, min_distance_target_goals, correlation, max_distance_goal, grid_table, target_unreached = get_experience(
            training_state,
            training_state.normalizer_params,
            training_state.policy_params,
            env_state,
            buffer_state,
            goal_selection_strategy,
            experience_key,
            mega_cutoff,
            False,
            ucb_params,
        )
        achievement_loss, achievement_params, achievement_optimizer_state = achievement_update(training_state.achievement_params, ucb_params, jnp.array([goal_achieval_rate]), optimizer_state=training_state.achievement_optimizer_state)
        ucb_params = adapt_ucb_params(training_state=training_state, goal_achieval_rate=goal_achieval_rate, ucb_params=ucb_params)
        training_state = training_state.replace(
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + env_steps_per_actor_step,
            achievement_optimizer_state=achievement_optimizer_state,
            achievement_params=achievement_params
        )

        training_state, buffer_state, metrics = train_steps(training_state, buffer_state, training_key)
        metrics["achievement_loss"] = achievement_loss
        return training_state, env_state, buffer_state, metrics, mega_cutoff, goal_achieval_rate, mean_ini_goal_mean, goals, min_distance_target_all, min_distance_target_goals, correlation, max_distance_goal, ucb_params, grid_table, target_unreached

    def prefill_replay_buffer(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:
        def f(carry, unused):
            del unused
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            new_normalizer_params, env_state, buffer_state, mega_cutoff, goal_achieval_rate, mean_ini_goal_mean, goals, min_distance_target_all, min_distance_target_goals, correlation, max_distance_goal, grid_table, target_unreached = get_experience(
                training_state,
                training_state.normalizer_params,
                training_state.policy_params,
                env_state,
                buffer_state,
                "HER",
                key,
                0,
                False,
                jnp.array([0.0,0.0,0.0,0.0]),
            )
            new_training_state = training_state.replace(
                normalizer_params=new_normalizer_params,
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )
            return (new_training_state, env_state, buffer_state, new_key), ()

        return jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            (),
            length=num_prefill_actor_steps,
        )[0]

    prefill_replay_buffer = jax.pmap(prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME)

    def train_steps(
        training_state: TrainingState,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, ReplayBufferState, Metrics]:
        experience_key, training_key, sampling_key = jax.random.split(key, 3)
        buffer_state, transitions = replay_buffer.sample(buffer_state)

        batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
        transitions = jax.vmap(TrajectoryUniformSamplingQueue.flatten_crl_fn, in_axes=(None, None, 0, 0))(
            config, env, transitions, batch_keys
        )

        # Shuffle transitions and reshape them into (number_of_sgd_steps, batch_size, ...)
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
            transitions,
        )
        permutation = jax.random.permutation(experience_key, len(transitions.observation))
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, batch_size) + x.shape[1:]),
            transitions,
        )

        (training_state, _), metrics = jax.lax.scan(update_step, (training_state, training_key), transitions)
        return training_state, buffer_state, metrics

    def scan_train_steps(n, ts, bs, update_key):

        def body(carry, unsued_t):
            ts, bs, update_key = carry
            new_key, update_key = jax.random.split(update_key)
            ts, bs, metrics = train_steps(ts, bs, update_key)
            return (ts, bs, new_key), metrics

        return jax.lax.scan(body, (ts, bs, update_key), (), length=n)

    def training_epoch(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
        mega_cutoff: int,
        ucb_params: jnp.array,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics, int, float]:
        def f(carry, unused_t):
            ts, es, bs, k, mega_cutoff, goal_achieval_rate, mean_ini_goal_mean, goals, min_distance_target_all, min_distance_target_goals, correlation, max_distance_goal, ucb_params, grid_table, target_unreached  = carry
            k, new_key, update_key = jax.random.split(k, 3)
            ts, es, bs, metrics, mega_cutoff, goal_achieval_rate, mean_ini_goal_mean, goals, min_distance_target_all, min_distance_target_goals, correlation, max_distance_goal, ucb_params, grid_table, target_unreached = training_step(ts, es, bs, k, mega_cutoff, ucb_params)
            (ts, bs, update_key), _ = scan_train_steps(train_step_multiplier - 1, ts, bs, update_key)
            return (ts, es, bs, new_key, mega_cutoff, goal_achieval_rate, mean_ini_goal_mean, goals, min_distance_target_all, min_distance_target_goals, correlation, max_distance_goal, ucb_params, grid_table, target_unreached), metrics
        insize = grid_size if grid_size > 0 else 1
        multiplier = 1 if len(env.goal_indices) == 2 else len(env.grid_heights)
        grid_table = jnp.zeros(shape=((insize**2)*multiplier, env.goal_indices.shape[0]+6)) if grid_size > 0 else 0
        (training_state, env_state, buffer_state, key, mega_cutoff, goal_achieval_rate, mean_ini_goal_mean, goals, min_distance_target_all, min_distance_target_goals, correlation, max_distance_goal, ucb_params, grid_table, target_unreached), metrics = jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key, mega_cutoff, 0, 0, jnp.zeros(shape=(num_envs, 2*env.goal_indices.shape[0]+6)), 0, 0, 0, 0, ucb_params, grid_table, 1),
            (),
            length=num_training_steps_per_epoch,
        )
        metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        return training_state, env_state, buffer_state, metrics, mega_cutoff, goal_achieval_rate, mean_ini_goal_mean, goals, min_distance_target_all, min_distance_target_goals, correlation, max_distance_goal, ucb_params, grid_table, target_unreached

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

    # Note that this is NOT a pure jittable method.
    def training_epoch_with_timing(
        training_state: TrainingState, env_state: envs.State, buffer_state: ReplayBufferState, key: PRNGKey, mega_cutoff: int, ucb_params: jnp.array
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics, int]:
        nonlocal training_walltime
        t = time.time()
        (training_state, env_state, buffer_state, metrics, mega_cutoff, goal_achieval, mean_ini_goal_mean, goals, min_distance_target_all, min_distance_target_goals, correlation, max_distance_goal, ucb_params, grid_table, target_unreached) = training_epoch(
            training_state, env_state, buffer_state, key, jnp.array([mega_cutoff]),jnp.array([ucb_params])
        )
        ucb_params = jnp.squeeze(ucb_params)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/mega_cutoff": jnp.mean(mega_cutoff),
            "training/goal_achieval": jnp.mean(goal_achieval),
            "training/sampled_goal_values": jnp.mean(mean_ini_goal_mean),
            "training/min_distance_target_all": jnp.mean(min_distance_target_all),
            "training/min_distance_target_goals": jnp.mean(min_distance_target_goals),
            "training/correlation_value_distance": jnp.mean(correlation),
            "training/max_distance_goal": jnp.mean(max_distance_goal),
            "training/target_unreached": jnp.mean(target_unreached),
            "training/walltime": training_walltime,
            "training/beta_mean_ini": jnp.mean(ucb_params[0]),
            "training/beta_mean_tar": jnp.mean(ucb_params[1]),
            "training/beta_std_ini": jnp.mean(ucb_params[2]),
            "training/beta_std_tar": jnp.mean(ucb_params[3]),
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        return training_state, env_state, buffer_state, metrics, jnp.mean(jnp.squeeze(mega_cutoff)), goals, ucb_params, grid_table  # pytype: disable=bad-return-type  # py311-upgrade

    global_key, local_key = jax.random.split(rng)
    local_key = jax.random.fold_in(local_key, process_id)

    # Training state init
    training_state = _init_training_state(
        key=global_key,
        obs_size=obs_size,
        local_devices_to_use=local_devices_to_use,
        td3_network=td3_network,
        policy_optimizer=policy_optimizer,
        alpha_optimizer=alpha_optimizer,
        q_optimizer=q_optimizer,
        metric_optimizer=metric_optimizer,
        achievement_optimizer=achievement_optimizer,
        dynamics_optimizer= dynamics_optimizer,
    )
    del global_key

    local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)

    # Env init
    env_keys = jax.random.split(env_key, num_envs // jax.process_count())
    env_keys = jnp.reshape(env_keys, (local_devices_to_use, -1) + env_keys.shape[1:])
    env_state = jax.pmap(env.reset)(env_keys)
    # Replay buffer init
    buffer_state = jax.pmap(replay_buffer.init)(jax.random.split(rb_key, local_devices_to_use))

    if not eval_env:
        eval_env = environment
    if randomization_fn is not None:
        v_randomization_fn = functools.partial(randomization_fn, rng=jax.random.split(eval_key, num_eval_envs))
    eval_env = TrajectoryIdWrapper(eval_env)
    eval_env = wrap_for_training(
        eval_env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )

    evaluator = CrlEvaluator(
        eval_env,
        functools.partial(make_policy, exploration_noise=0, noise_clip=0, deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )

    # Run initial eval
    metrics = {}
    if process_id == 0 and num_evals > 1:
        metrics = evaluator.run_evaluation(
            _unpmap((training_state.normalizer_params, training_state.policy_params)), training_metrics={}
        )
        logging.info(metrics)
        progress_fn(0, metrics, make_policy, _unpmap((training_state.normalizer_params, training_state.policy_params)), unwrapped_env)

    # Create and initialize the replay buffer.
    t = time.time()
    prefill_key, local_key = jax.random.split(local_key)
    prefill_keys = jax.random.split(prefill_key, local_devices_to_use)
    training_state, env_state, buffer_state, _ = prefill_replay_buffer(
        training_state, env_state, buffer_state, prefill_keys
    )
    ucb_params = jnp.array([ucb_mean_ini, ucb_mean_tar, ucb_std_ini, ucb_std_tar])
    replay_size = jnp.sum(jax.vmap(replay_buffer.size)(buffer_state)) * jax.process_count()
    logging.info("replay size after prefill %s", replay_size)
    assert replay_size >= min_replay_size
    training_walltime = time.time() - t

    current_step = 0
    count = 0
    mega_cutoff = -1 * negative_rewards * mega_min_val + (1-negative_rewards) * mega_min_val
    for eval_epoch_num in range(num_evals_after_init):
        logging.info("step %s", current_step)

        # Optimization
        epoch_key, local_key = jax.random.split(local_key)
        epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
        (training_state, env_state, buffer_state, training_metrics, mega_cutoff, goals, ucb_params, grid_table_goals) = training_epoch_with_timing(
            training_state, env_state, buffer_state, epoch_keys, mega_cutoff, ucb_params
        )
        if log_wandb:
            count += 1
            if count == goal_log_delay:
                step_arr = jnp.ones(shape=(num_envs,1)) * current_step
                goals = jnp.concatenate([step_arr, goals[0]], axis=1)
                goal_table = wandb.Table(columns=["step"] + [f"goal_{i}" for i in range(len(env.goal_indices))] + ["mean_val", "std_val", "mean_val_tar", "std_val_tar", "unachieved"] + [f"random_state_{i}" for i in range(len(env.goal_indices))] + ["target_unreached"], data=list(goals))
                count = 0
                wandb.log({"goals": goal_table})
                if grid_size > 0:
                    step_arr = jnp.ones(shape=(grid_table_goals[0].shape[0],1)) * current_step
                    grid_table_goals = jnp.concatenate([step_arr, grid_table_goals[0]], axis=1)
                    grid_table = wandb.Table(columns=["step"] + [f"goal_{i}" for i in range(len(env.goal_indices))] + ["mean_val", "std_val", "mean_val_tar", "std_val_tar"] + ["goal_bin_1", "goal_bin_2"], data=list(grid_table_goals))
                    wandb.log({"grid_goals": grid_table})
        current_step = int(_unpmap(training_state.env_steps))
        # Eval and logging
        if process_id == 0:
            if checkpoint_logdir and eval_epoch_num % 10 == 0: # Only log model every 10th evaluation
                # Save current policy.
                params = _unpmap((training_state.normalizer_params, training_state.policy_params, training_state.q_params, training_state.metric_params))
                path = f"{checkpoint_logdir}_td3_{current_step}.pkl"
                model.save_params(path, params)

            # Run evals.
            if eval_target_policy:
                metrics = evaluator.run_evaluation(
                    _unpmap((training_state.normalizer_params, training_state.slow_target_policy_params)), training_metrics
                )
            else:
                metrics = evaluator.run_evaluation(
                    _unpmap((training_state.normalizer_params, training_state.policy_params)), training_metrics
                )
            
            logging.info(metrics)
            do_render = (eval_epoch_num % visualization_interval) == 0
            progress_fn(current_step, metrics, make_policy, _unpmap((training_state.normalizer_params, training_state.policy_params)), unwrapped_env, do_render)


    total_steps = current_step
    assert total_steps >= num_timesteps

    params = _unpmap((training_state.normalizer_params, training_state.policy_params, training_state.q_params))

    # If there was no mistakes the training_state should still be identical on all
    # devices.
    pmap.assert_is_replicated(training_state)
    logging.info("total steps: %s", total_steps)
    pmap.synchronize_hosts()
    return (make_policy, params, metrics)
