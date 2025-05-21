import os
from typing import Tuple

from brax import base
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
from jax import numpy as jnp
import mujoco
import numpy as np
import pickle


class HighDimensionalMaze(PipelineEnv):
    def __init__(
        self,
        file_name,
        backend="generalized",
        **kwargs,
    ):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", "ant.xml")
        sys = mjcf.load(path)
        n_frames = 5
        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({"opt.timestep": 0.0025}) # 0.005
            n_frames = 10

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)
        #file_name = '~/Developer/information-directed-gcrl-jax/maze_layouts/2d_test.pkl'
        file_name = os.path.expanduser(file_name)
        file = open(file_name,'rb')
        dic = pickle.load(file)
        self.walls = jnp.array(dic["walls"])
        self.n_dim = dic["n_dim"]
        self.radius = dic["radius"]
        self.goal_indices = jnp.linspace(0,self.n_dim-1, self.n_dim).astype(int)
        self.goal_indices_2 = self.goal_indices
        jax.debug.print("goal ind {}", self.goal_indices)
        self.state_dim = self.n_dim
        self.step_length = 0.5
        self.goal_reach_thresh = self.step_length
        self._reset_noise_scale = 0.1
        self.grid_heights = [0]
        self.max_dis = self.radius * self.n_dim

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""

        rng, rng1, rng2 = jax.random.split(rng, 3)
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        rand_unif = jax.random.uniform(rng, (self.sys.q_size(),), minval=low, maxval=hi)

        q = rand_unif
        qd = self.sys.init_q*0

        target = jnp.ones(self.n_dim) * self.radius
        q = q.at[-self.n_dim:].set(target)
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)

        reward, done, zero = jnp.zeros(3)
        metrics = {
            "reward_forward": zero,
            "reward_survive": zero,
            "reward_ctrl": zero,
            "reward_contact": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "forward_reward": zero,
            "dist": zero,
            "success": zero,
            "success_easy": zero,
        }
        state = State(pipeline_state, obs, reward, done, metrics)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        q = pipeline_state0.q
        qd = pipeline_state0.qd
        end_location = q[:self.n_dim] + self.step_length * action
        end_index = end_location.astype(int)+self.radius+1
        is_wall = self.walls[tuple(end_index)]
        q = q.at[:self.n_dim].set(q[:self.n_dim] * is_wall + (1-is_wall) * end_location)
        target = q[-self.n_dim:]
        #pipeline_state = self.pipeline_init(q, qd)
        pipeline_state = pipeline_state0.replace(q=q)
        obs = self._get_obs(pipeline_state)
        done = 0.0
        dist = jnp.linalg.norm(q[:self.n_dim]-target)
        success = jnp.array(dist < self.goal_reach_thresh, dtype=float)
        reward = success
        state.metrics.update(
            reward_forward=0.0,
            reward_survive=0.0,
            reward_ctrl=0.0,
            reward_contact=0.0,
            x_position=pipeline_state.x.pos[0, 0],
            y_position=pipeline_state.x.pos[0, 1],
            distance_from_origin=0.0,
            x_velocity=0.0,
            y_velocity=0.0,
            forward_reward=0.0,
            dist=0.0,
            success=success,
            success_easy=0.0,
        )
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe ant body position and velocities."""
        # remove target q, qd
        qpos = pipeline_state.q[:self.n_dim]
        #qvel = pipeline_state.qd[:-2]

        target_pos = pipeline_state.q[-self.n_dim:]

        return jnp.concat([qpos, target_pos])
    
    @property
    def action_size(self) -> int:
        return self.n_dim
    
    @property
    def observation_size(self) -> int:
        return 2*self.n_dim