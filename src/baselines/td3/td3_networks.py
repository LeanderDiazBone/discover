"""TD3 networks."""

from typing import Sequence, Tuple

import jax.numpy as jnp
from brax.training import networks
from brax.training import types
from brax.training.networks import ActivationFn, FeedForwardNetwork, Initializer, MLP
from brax.training.types import PRNGKey
from flax import linen, struct
from brax.training import distribution
import jax


@struct.dataclass
class Networks:
    policy_network: networks.FeedForwardNetwork
    q_network: networks.FeedForwardNetwork
    metric_network: networks.FeedForwardNetwork
    achievement_network: networks.FeedForwardNetwork
    dynamics_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def make_inference_fn(algo_networks: Networks, algo = "TD3"):
    """Creates params and inference function for the TD3 agent."""

    def make_policy_td3(params: types.PolicyParams, exploration_noise=0.0, noise_clip=0.0, deterministic=False) -> types.Policy:
        def policy(observations: types.Observation,
                   key_noise: PRNGKey) -> Tuple[types.Action, types.Extra]:
            actions = algo_networks.policy_network.apply(*params, observations)
            noise = (jax.random.normal(key_noise, actions.shape) * exploration_noise).clip(-noise_clip, noise_clip)
            return actions + noise, {}

        return policy

    def make_policy_sac(
        params: types.PolicyParams, exploration_noise=0.0, noise_clip=0.0, deterministic: bool = False
        ) -> types.Policy:

        def policy(
            observations: types.Observation, key_sample: PRNGKey
        ) -> Tuple[types.Action, types.Extra]:
            logits = algo_networks.policy_network.apply(*params, observations)
            if deterministic:
                return algo_networks.parametric_action_distribution.mode(logits), {}
            return (algo_networks.parametric_action_distribution.sample(logits, key_sample),{},)
        return policy

    if algo in ["TD3", "ThompTD3"]:
        return make_policy_td3
    elif algo in ["SAC", "MaxInfoSAC"]:
        return make_policy_sac
    else:
        raise NotImplementedError(f"The algorithm {algo} is not implemented.")


def make_policy_network(
        param_size: int,
        obs_size: int,
        preprocess_observations_fn: types.PreprocessObservationFn = types
        .identity_observation_preprocessor,
        hidden_layer_sizes: Sequence[int] = (256, 256),
        activation: ActivationFn = linen.relu,
        kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
        layer_norm: bool = False) -> FeedForwardNetwork:
    """Creates a policy network."""
    policy_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [param_size],
        activation=activation,
        kernel_init=kernel_init,
        layer_norm=layer_norm)

    def apply(processor_params, policy_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        raw_actions = policy_module.apply(policy_params, obs)
        return linen.tanh(raw_actions)

    dummy_obs = jnp.zeros((1, obs_size))
    return FeedForwardNetwork(init=lambda key: policy_module.init(key, dummy_obs), apply=apply)

class MLP_CUSTOM(linen.Module):
  """MLP module."""

  layer_sizes: Sequence[int]
  activation: ActivationFn = linen.relu
  activation_final: ActivationFn = linen.relu
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
  activate_final: bool = False
  bias: bool = True
  layer_norm: bool = False

  @linen.compact
  def __call__(self, data: jnp.ndarray):
    hidden = data
    for i, hidden_size in enumerate(self.layer_sizes):
        hidden = linen.Dense(
            hidden_size,
            name=f'hidden_{i}',
            kernel_init=self.kernel_init,
            use_bias=self.bias,
        )(hidden)
        if i != len(self.layer_sizes) - 1:
            hidden = self.activation(hidden)
            if self.layer_norm:
                hidden = linen.LayerNorm()(hidden)
        if i == len(self.layer_sizes) - 1 and self.activate_final:
            hidden = self.activation_final(hidden)
            if self.layer_norm:
                hidden = linen.LayerNorm()(hidden)
    return hidden

def make_custom_q_network(
    obs_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activate_final: bool = False,
    activation: ActivationFn = linen.relu,
    activation_final: ActivationFn = linen.softplus,
    n_critics: int = 2,
    layer_norm: bool = False,
) -> FeedForwardNetwork:
  """Creates a value network."""

  class QModule(linen.Module):
    """Q Module."""

    n_critics: int

    @linen.compact
    def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
      hidden = jnp.concatenate([obs, actions], axis=-1)
      res = []
      for _ in range(self.n_critics):
        q = MLP_CUSTOM(
            layer_sizes=list(hidden_layer_sizes) + [1],
            activation=activation,
            activation_final=activation_final,
            activate_final=activate_final,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            layer_norm=layer_norm,
        )(hidden)
        res.append(q)
      return jnp.concatenate(res, axis=-1)

  q_module = QModule(n_critics=n_critics)

  def apply(processor_params, q_params, obs, actions):
    obs = preprocess_observations_fn(obs, processor_params)
    return q_module.apply(q_params, obs, actions)

  dummy_obs = jnp.zeros((1, obs_size))
  dummy_action = jnp.zeros((1, action_size))
  return FeedForwardNetwork(
      init=lambda key: q_module.init(key, dummy_obs, dummy_action), apply=apply
  )

def make_custom_dynamics_network(
    obs_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    n_ensemble: int = 2,
    layer_norm: bool = False,
) -> FeedForwardNetwork:
  """Creates a value network."""

  class DynamicsModule(linen.Module):
    """Q Module."""

    n_ensemble: int

    @linen.compact
    def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
      hidden = jnp.concatenate([obs, actions], axis=-1)
      res = []
      for _ in range(self.n_ensemble):
        q = MLP_CUSTOM(
            layer_sizes=list(hidden_layer_sizes) + [obs_size],
            activation=activation,
            activation_final=None,
            activate_final=None,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            layer_norm=layer_norm,
        )(hidden)
        res.append(q)
      return jnp.stack(res, axis=0)

  dynamics_module = DynamicsModule(n_ensemble=n_ensemble)

  def apply(processor_params, dynamics_params, obs, actions):
    obs = preprocess_observations_fn(obs, processor_params)
    return dynamics_module.apply(dynamics_params, obs, actions)

  dummy_obs = jnp.zeros((1, obs_size))
  dummy_action = jnp.zeros((1, action_size))
  return FeedForwardNetwork(
      init=lambda key: dynamics_module.init(key, dummy_obs, dummy_action), apply=apply
  )

def make_custom_achievement_predictor_network(
    num_params: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    activate_final: bool = True,
    activation_final: ActivationFn = linen.sigmoid,
) -> FeedForwardNetwork:
  """Creates a value network."""

  class AchiementModule(linen.Module):
    """Achievement Module."""

    num_params: int

    @linen.compact
    def __call__(self, params: jnp.ndarray):
      q = MLP_CUSTOM(
        layer_sizes=[1],
        activation=None,
        activation_final=activation_final,
        activate_final=activate_final,
        kernel_init=jax.nn.initializers.lecun_uniform(),
      )(params)
      return q

  achievement_module = AchiementModule(num_params=num_params)

  def apply(ach_params, params):
    return achievement_module.apply(ach_params, params)

  dummy_params = jnp.zeros((1, num_params))
  return FeedForwardNetwork(
      init=lambda key: achievement_module.init(key, dummy_params), apply=apply
  )

def make_td3_networks(
        observation_size: int,
        action_size: int,
        preprocess_observations_fn: types.PreprocessObservationFn = types
        .identity_observation_preprocessor,
        hidden_layer_sizes: Sequence[int] = (256, 256),
        activate_final: bool = False,
        activation: networks.ActivationFn = linen.relu,
        activation_final: ActivationFn = linen.softplus,
        n_critics: int = 2,
        param_size: int = 4,
        algo = "TD3") -> Networks:
    """Make TD3 networks."""
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    if algo in ["TD3", "ThompTD3"]:
        policy_network = make_policy_network(
            action_size,
            observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
        )
    elif algo in ["SAC", "MaxInfoSAC"]:
        policy_network = networks.make_policy_network(
            parametric_action_distribution.param_size,
            observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            layer_norm=False,
        )
    else:
       raise NotImplementedError(f"The algorithm {algo} is not implemented.")
        # q_network = networks.make_q_network(
        #     observation_size,
        #     action_size,
        #     preprocess_observations_fn=preprocess_observations_fn,
        #     hidden_layer_sizes=hidden_layer_sizes,
        #     activation=activation,
        #     layer_norm=False,
        # )
    q_network = make_custom_q_network(
            observation_size,
            action_size,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layer_sizes=hidden_layer_sizes,
            activate_final=activate_final,
            activation_final=activation_final,
            activation=activation,
            n_critics = n_critics)
    metric_network = make_custom_q_network(
        observation_size,
        action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=(20,),
        activate_final=False,
        activation_final=activation_final,
        activation=linen.relu,
        n_critics=2)
    
    achievement_network = make_custom_achievement_predictor_network(
        num_params=param_size, 
        preprocess_observations_fn=preprocess_observations_fn)
    
    dynamics_network = make_custom_dynamics_network(
        observation_size,
        action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        n_ensemble = n_critics)

    return Networks(
        policy_network=policy_network,
        q_network=q_network,
        metric_network=metric_network,
        achievement_network=achievement_network,
        dynamics_network=dynamics_network,
        parametric_action_distribution=parametric_action_distribution)

