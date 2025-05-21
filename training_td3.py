import functools
import json
import os
import pickle

import wandb
from brax.io import model
from pyinstrument import Profiler

from src.baselines.td3.td3_train import train
from utils import MetricsRecorder, get_env_config, create_env, create_eval_env, create_parser


def main(args):
    """
    Main function orchestrating the overall setup, initialization, and execution
    of training and evaluation processes. This function performs the following:
    1. Environment setup
    2. Directory creation for logging and checkpoints
    3. Training function creation
    4. Metrics recording
    5. Progress logging and monitoring
    6. Model saving and inference

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments specifying configuration parameters for the
        training and evaluation processes.

    """
    env = create_env(**vars(args), file_name=args.env_file_name)
    eval_env = create_eval_env(args)
    config = get_env_config(args)

    os.makedirs('./runs', exist_ok=True)
    run_dir = './runs/run_{name}_s_{seed}'.format(name=args.exp_name, seed=args.seed)
    ckpt_dir = run_dir + '/ckpt'
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(run_dir + '/args.pkl', 'wb') as f:
        pickle.dump(args, f)

    train_fn = functools.partial(
        train,
        num_timesteps=args.num_timesteps,
        num_evals=args.num_evals,
        reward_scaling=1,
        episode_length=args.episode_length,
        normalize_observations=False,
        action_repeat=args.action_repeat,
        discounting=args.discounting,
        learning_rate=args.critic_lr,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        unroll_length=args.unroll_length,
        max_devices_per_host=1,
        max_replay_size=args.max_replay_size,
        min_replay_size=args.min_replay_size,
        seed=args.seed,
        eval_env=eval_env,
        config=config,
        goal_selection_strategy=args.goal_selection_strategy,
        ensemble_size=args.ensemble_size,
        filter_goals=args.filter_goals,
        ucb_mean_ini= args.ucb_mean_ini,
        ucb_mean_tar= args.ucb_mean_tar,
        ucb_std_ini= args.ucb_std_ini,
        ucb_std_tar= args.ucb_std_tar,
        negative_rewards= args.negative_rewards,
        num_goals=args.num_goals,
        mega_cutoff_step_size = args.mega_cutoff_step_size,
        manipulation = args.manipulation,
        train_step_multiplier = args.train_step_multiplier,
        log_wandb = args.log_wandb,
        tau = args.tau,
        tau_policy = args.tau_policy,
        deterministic_eval = args.deterministic_eval,
        policy_delay = args.policy_delay,
        activate_final = args.activate_final,
        target_computation = args.target_computation,
        transform_ucb = args.transform_ucb,
        standardize_ucb = args.standardize_ucb,
        critic_prior_file_name = args.critic_prior_file_name,
        save_models = args.save_models,
        ckpt_dir = ckpt_dir+"/",
        obs_size_prior = args.obs_size_prior,
        action_size_prior = args.action_size_prior,
        use_metric = args.use_metric,
        eval_target_policy = args.eval_target_policy,
        adaptation_strategy = args.adaptation_strategy,
        goal_log_delay = args.goal_log_delay,
        continue_strategy = args.continue_strategy,
        grid_size = args.grid_size,
        adaptation_rate = args.adaptation_rate,
        goal_achievement_target = args.goal_achievement_target,
        dir_var = args.dir_var,
        cur_alpha = args.cur_alpha,
        exploration_noise = args.exploration_noise,
        algo = args.algo,
    )

    metrics_to_collect = [
        "eval/episode_reward",
        "eval/episode_success",
        "eval/episode_success_any",
        "eval/episode_success_hard",
        "eval/episode_success_easy",
        "eval/episode_reward_dist",
        "eval/episode_reward_near",
        #"eval/episode_reward_ctrl",
        #"eval/episode_dist",
        "eval/episode_reward_survive",
        "training/actor_loss",
        "training/critic_loss",
        "training/sps",
        "training/entropy",
        "training/alpha",
        "training/alpha_loss",
        "training/dynamics_loss",
        "training/entropy",
        "training/mega_cutoff",
        "training/goal_achieval",
        "training/sampled_goal_values",
        "training/min_distance_target_all",
        "training/min_distance_target_goals",
        "training/correlation_value_distance",
        "training/max_distance_goal",
        "training/beta_mean_ini",
        "training/beta_mean_tar",
        "training/beta_std_ini",
        "training/beta_std_tar",
        "training/achievement_loss",
        "training/target_unreached",
    ]

    metrics_recorder = MetricsRecorder(args.num_timesteps, metrics_to_collect, run_dir, args.exp_name, args.render_delay)

    make_policy, params, _ = train_fn(environment=env, progress_fn=metrics_recorder.progress)
    model.save_params(ckpt_dir + '/final', params)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    print("Arguments:")
    print(
        json.dumps(
            vars(args), sort_keys=True, indent=4
        )
    )
    utd_ratio = (
        args.num_envs
        * args.episode_length
        * args.train_step_multiplier
        / args.batch_size
    ) / (args.num_envs * args.unroll_length)
    print(f"Updates per environment step: {utd_ratio}")
    args.utd_ratio = utd_ratio

    wandb.init(
        project=args.project_name,
        group=args.group_name,
        name=args.exp_name,
        config=vars(args),
        mode="online" if args.log_wandb else "disabled",
    )

    with Profiler(interval=0.1) as profiler:
        main(args)
    profiler.print()
    profiler.open_in_browser()
