"""Twin Delayed Deep Deterministic Policy Gradient (TD3) losses.

See: https://arxiv.org/pdf/1802.09477.pdf
"""
from typing import Any

import jax
import jax.numpy as jnp
from brax.training import types
from brax.training.types import Params
from brax.training.types import PRNGKey
from src.baselines.td3 import td3_networks

Transition = types.Transition


def make_losses(
        td3_network: td3_networks.Networks,
        reward_scaling: float,
        discounting: float,
        smoothing: float,
        noise_clip: float,
        action_size: int,
        max_action: float = 1.0,
        bc: bool = False,
        alpha: float = 2.5,
        cur_alpha: float = 0.0,
        negative_rewards: bool = False,
        target_computation: str = "min",
        n_critics = 6,
        algo = "TD3",
        ):
    """Creates the TD3 losses."""
    policy_network = td3_network.policy_network
    q_network = td3_network.q_network
    metric_network = td3_network.metric_network
    achievement_network = td3_network.achievement_network
    dynamics_network = td3_network.dynamics_network
    parametric_action_distribution = td3_network.parametric_action_distribution
    target_entropy = -0.5 * action_size

    def metric_loss(q_params: Params,
            target_q_params: Params,
            target_policy_params: Params,
            normalizer_params: Any,
            transitions: Transition,
            key: PRNGKey,) -> jnp.ndarray:
        """Calculates the TD3 critic loss for metric."""
        current_q = metric_network.apply(normalizer_params, q_params, transitions.observation, transitions.action)
        next_actions = policy_network.apply(normalizer_params, target_policy_params, transitions.next_observation)
        smoothing_noise = (jax.random.normal(key, next_actions.shape) * smoothing).clip(-noise_clip, noise_clip)
        next_actions = (next_actions + smoothing_noise).clip(-max_action, max_action)
        next_q = metric_network.apply(normalizer_params, target_q_params, transitions.next_observation, next_actions)
        if target_computation == "mean":
            target_q = jnp.mean(next_q, axis=-1)
        elif target_computation == "min":
            target_q = jnp.min(next_q, axis=-1)
        elif target_computation == "min_random":
            key, next_key = jax.random.split(key)
            rdm_idx = jax.random.choice(key, jnp.linspace(0, next_q.shape[-1]-1,  next_q.shape[-1]), shape=(2,), replace = False).astype(int) #int(next_q.shape[-1]/2)
            target_q = jnp.min(next_q[:,rdm_idx], axis=-1)
        elif target_computation == "single":
            target_q = next_q
        
        if target_computation == "single":
            target_q = jax.lax.stop_gradient(
                jnp.expand_dims(transitions.reward, -1) * reward_scaling +jnp.expand_dims(transitions.discount, -1)  * discounting * target_q)
        else:
            target_q = jax.lax.stop_gradient(
                transitions.reward * reward_scaling + transitions.discount * discounting * target_q)
        if target_computation == "single":
            q_error = current_q - target_q
        else:
            q_error = current_q - jnp.expand_dims(target_q, -1)
        q_loss = 0.5 * jnp.mean(jnp.square(q_error))

        return q_loss
    
    def dynamics_loss(dynamics_params: Params,
            normalizer_params: Any,
            transitions: Transition,
            key: PRNGKey,) -> jnp.ndarray:
        """Calculates the dynamics model loss."""

        dynamics_prediction = dynamics_network.apply(normalizer_params, dynamics_params, transitions.observation, transitions.action)
        dynamics_loss = jnp.mean(jnp.linalg.norm(dynamics_prediction - transitions.next_observation, axis=0))
        return dynamics_loss


    def alpha_loss(
      log_alpha: jnp.ndarray,
      policy_params: Params,
      normalizer_params: Any,
      transitions: Transition,
      key: PRNGKey,
  ) -> jnp.ndarray:
        """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
        dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.observation
        )
        action = parametric_action_distribution.sample_no_postprocessing(
            dist_params, key
        )
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        alpha = jnp.exp(log_alpha)
        alpha_loss = alpha * jax.lax.stop_gradient(-log_prob - target_entropy)
        return jnp.mean(alpha_loss)
    
    def alpha_loss_td3(
      log_alpha: jnp.ndarray,
      policy_params: Params,
      normalizer_params: Any,
      transitions: Transition,
      key: PRNGKey,
  ) -> jnp.ndarray:
        alpha = jnp.exp(log_alpha)
        alpha_loss = alpha
        return jnp.mean(alpha_loss)

    def critic_loss_td3(
            q_params: Params,
            target_q_params: Params,
            target_policy_params: Params,
            normalizer_params: Any,
            dynamics_params: Params,
            alpha: jnp.ndarray, # For consistency (not used)
            transitions: Transition,
            key: PRNGKey,) -> jnp.ndarray:
        """Calculates the TD3 critic loss."""
        current_q = q_network.apply(normalizer_params, q_params, transitions.observation, transitions.action)
        next_actions = policy_network.apply(normalizer_params, target_policy_params, transitions.next_observation)
        smoothing_noise = (jax.random.normal(key, next_actions.shape) * smoothing).clip(-noise_clip, noise_clip)
        next_actions = (next_actions + smoothing_noise).clip(-max_action, max_action)
        next_q = q_network.apply(normalizer_params, target_q_params, transitions.next_observation, next_actions)
        if target_computation == "mean":
            target_q = jnp.mean(next_q, axis=-1)
        elif target_computation == "min":
            target_q = jnp.min(next_q, axis=-1)
        elif target_computation == "min_random":
            key, next_key = jax.random.split(key)
            rdm_idx = jax.random.choice(key, jnp.linspace(0, next_q.shape[-1]-1,  next_q.shape[-1]), shape=(2,), replace = False).astype(int) #int(next_q.shape[-1]/2)
            target_q = jnp.min(next_q[:,rdm_idx], axis=-1)
        elif target_computation == "single":
            target_q = next_q
        
        if target_computation == "single":
            target_q = jax.lax.stop_gradient(
                jnp.expand_dims(transitions.reward, -1) * reward_scaling +jnp.expand_dims(transitions.discount, -1)  * discounting * target_q)
        else:
            target_q = jax.lax.stop_gradient(
                transitions.reward * reward_scaling + transitions.discount * discounting * target_q)
        #jax.debug.print("target_q: {}", target_q[:10])
        if target_computation == "single":
            q_error = current_q - target_q
        else:
            q_error = current_q - jnp.expand_dims(target_q, -1)
        q_loss = 0.5 * jnp.mean(jnp.square(q_error))

        return q_loss
    
    def actor_loss_td3(
        policy_params: Params,
        q_params: Params,
        normalizer_params: Any,
        dynamics_params: Params,
        alpha: jnp.ndarray,
        transitions: Transition,
        key: PRNGKey,) -> jnp.ndarray:
        """Calculates the TD3 actor loss."""

        new_actions = policy_network.apply(normalizer_params, policy_params, transitions.observation)
        q_new_actions = q_network.apply(normalizer_params, q_params, transitions.observation, new_actions)
        q_new_actions, _ = jnp.split(q_new_actions, 2, axis=-1)
        q_mean = jnp.mean(q_new_actions)
        q_std = jnp.mean(jnp.nan_to_num(jnp.std(q_new_actions, axis=-1)))
        if cur_alpha:
            return - q_mean - cur_alpha * q_std 
        else:
            return - q_mean 
        
    def actor_loss_td3_thompson(
        policy_params: Params,
        q_params: Params,
        normalizer_params: Any,
        dynamics_params: Params,
        alpha: jnp.ndarray,
        transitions: Transition,
        key: PRNGKey,) -> jnp.ndarray:
        """Calculates the TD3 actor loss."""

        new_actions = policy_network.apply(normalizer_params, policy_params, transitions.observation)
        q_new_actions = q_network.apply(normalizer_params, q_params, transitions.observation, new_actions)
        rand_critic = jax.random.randint(key, minval=0, maxval=n_critics, shape=(1,))
        rdm_idx = jax.random.choice(key, jnp.linspace(0, n_critics-1,  n_critics), shape=(1,), replace = False).astype(int) #int(next_q.shape[-1]/2)

        q_new_actions_split = jnp.stack(jnp.split(q_new_actions, n_critics, axis=-1), axis=0)
        q_mean = jnp.mean(q_new_actions_split[rdm_idx])
        return - q_mean

    def critic_loss_sac(
        q_params: Params,
        target_q_params: Params,
        policy_params: Params,
        normalizer_params: Any,
        dynamics_params: Params,
        alpha: jnp.ndarray,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        q_old_action = q_network.apply(
            normalizer_params, q_params, transitions.observation, transitions.action
        )
        next_dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.next_observation
        )
        next_action = parametric_action_distribution.sample_no_postprocessing(
            next_dist_params, key
        )
        next_log_prob = parametric_action_distribution.log_prob(
            next_dist_params, next_action
        )
        next_action = parametric_action_distribution.postprocess(next_action)
        next_q = q_network.apply(
            normalizer_params,
            target_q_params,
            transitions.next_observation,
            next_action,
        )
        if target_computation == "min_random":
            key, next_key = jax.random.split(key)
            rdm_idx = jax.random.choice(key, jnp.linspace(0, next_q.shape[-1]-1,  next_q.shape[-1]), shape=(2,), replace = False).astype(int) #int(next_q.shape[-1]/2)
            tar_q = jnp.min(next_q[:,rdm_idx], axis=-1)
        else:
            tar_q = jnp.min(next_q, axis=-1)

        next_v = tar_q - alpha * next_log_prob
        target_q = jax.lax.stop_gradient(
            transitions.reward * reward_scaling
            + transitions.discount * discounting * next_v
        )
        q_error = q_old_action - jnp.expand_dims(target_q, -1)

        # Better bootstrapping for truncated episodes.
        truncation = transitions.extras['state_extras']['truncation']
        q_error *= jnp.expand_dims(1 - truncation, -1)

        q_loss = 0.5 * jnp.mean(jnp.square(q_error))
        return q_loss

    def actor_loss_sac(
        policy_params: Params,
        q_params: Params,
        normalizer_params: Any,
        dynamics_params: Params,
        alpha: jnp.ndarray,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.observation
        )
        action = parametric_action_distribution.sample_no_postprocessing(
            dist_params, key
        )
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        action = parametric_action_distribution.postprocess(action)
        q_action = q_network.apply(
            normalizer_params, q_params, transitions.observation, action
        )
        if target_computation == "min_random":
            key, next_key = jax.random.split(key)
            rdm_idx = jax.random.choice(key, jnp.linspace(0, q_action.shape[-1]-1,  q_action.shape[-1]), shape=(2,), replace = False).astype(int) #int(next_q.shape[-1]/2)
            min_q = jnp.min(q_action[:,rdm_idx], axis=-1)
        else:
            min_q = jnp.min(q_action, axis=-1)
        actor_loss = alpha * log_prob - min_q
        return jnp.mean(actor_loss)
    
    def critic_loss_max_info_sac(
        q_params: Params,
        target_q_params: Params,
        policy_params: Params,
        normalizer_params: Any,
        dynamics_params: Params,
        alpha: jnp.ndarray,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        q_old_action = q_network.apply(
            normalizer_params, q_params, transitions.observation, transitions.action
        )
        next_dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.next_observation
        )
        next_action = parametric_action_distribution.sample_no_postprocessing(
            next_dist_params, key
        )
        next_log_prob = parametric_action_distribution.log_prob(
            next_dist_params, next_action
        )
        next_action = parametric_action_distribution.postprocess(next_action)
        next_q = q_network.apply(
            normalizer_params,
            target_q_params,
            transitions.next_observation,
            next_action,
        )
        alpha_info = 1
        dynamics_prediction = dynamics_network.apply(normalizer_params, dynamics_params, transitions.next_observation, next_action)
        disagreement = jnp.std(dynamics_prediction, axis=0)
        unc_al = 1
        next_info = jnp.sum(jnp.log(jnp.square(disagreement)/unc_al + 1))
        if target_computation == "min_random":
            key, next_key = jax.random.split(key)
            rdm_idx = jax.random.choice(key, jnp.linspace(0, next_q.shape[-1]-1,  next_q.shape[-1]), shape=(2,), replace = False).astype(int) #int(next_q.shape[-1]/2)
            tar_q = jnp.min(next_q[:,rdm_idx], axis=-1)
        else:
            tar_q = jnp.min(next_q, axis=-1)
        next_v = tar_q - alpha * next_log_prob - alpha_info * next_info
        target_q = jax.lax.stop_gradient(
            transitions.reward * reward_scaling
            + transitions.discount * discounting * next_v
        )
        q_error = q_old_action - jnp.expand_dims(target_q, -1)

        # Better bootstrapping for truncated episodes.
        truncation = transitions.extras['state_extras']['truncation']
        q_error *= jnp.expand_dims(1 - truncation, -1)

        q_loss = 0.5 * jnp.mean(jnp.square(q_error))
        return q_loss


    def actor_loss_max_info_sac(
        policy_params: Params,
        q_params: Params,
        normalizer_params: Any,
        dynamics_params: Params,
        alpha: jnp.ndarray,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.observation
        )
        action = parametric_action_distribution.sample_no_postprocessing(
            dist_params, key
        )
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        action = parametric_action_distribution.postprocess(action)
        q_action = q_network.apply(
            normalizer_params, q_params, transitions.observation, action
        )
        if target_computation == "min_random":
            key, next_key = jax.random.split(key)
            rdm_idx = jax.random.choice(key, jnp.linspace(0, q_action.shape[-1]-1,  q_action.shape[-1]), shape=(2,), replace = False).astype(int) #int(next_q.shape[-1]/2)
            min_q = jnp.min(q_action[:,rdm_idx], axis=-1)
        else:
            min_q = jnp.min(q_action, axis=-1)
        alpha_info = 1
        dynamics_prediction = dynamics_network.apply(normalizer_params, dynamics_params, transitions.observation, transitions.action)
        disagreement = jnp.std(dynamics_prediction, axis=0)
        unc_al = 1
        info = jnp.sum(jnp.log(jnp.square(disagreement)/unc_al + 1))
        actor_loss = alpha * log_prob - alpha_info * info - min_q 
        return jnp.mean(actor_loss)


    def mean_squared_error(predictions, targets):
        return jnp.mean(jnp.square(predictions - targets))
    
    def achievement_loss(
            achievement_params: Params, 
            params: jnp.array,
            achievement: jnp.array) -> jnp.ndarray:
        pred_ach = achievement_network.apply(achievement_params, params=params)
        return mean_squared_error(pred_ach, achievement)

    if algo == "TD3":
        return critic_loss_td3, metric_loss, actor_loss_td3, achievement_loss, dynamics_loss, alpha_loss_td3
    elif algo == "SAC":
        return critic_loss_sac, metric_loss, actor_loss_sac, achievement_loss, dynamics_loss, alpha_loss
    elif algo == "MaxInfoSAC":
        return critic_loss_max_info_sac, metric_loss, actor_loss_max_info_sac, achievement_loss, dynamics_loss, alpha_loss
    elif algo == "ThompTD3":
        return critic_loss_td3, metric_loss, actor_loss_td3_thompson, achievement_loss, dynamics_loss, alpha_loss_td3
    else:
        raise NotImplementedError(f"The algorithm {algo} is not supported.")
