POINT_MAZE="--env_name=high_dimensional_maze --env_file_name=~/information-directed-gcrl-jax/maze_layouts/rand_4d_5h.pkl --num_timesteps=50000000 --episode_length=100 --num_envs=64 --train_step_multiplier=16 --num_goals=5000 --unroll_length=100 --batch_size=64"
DISCOVER="--algo=TD3 --goal_selection_strategy=UCB --ucb_std_tar=0.0 --ucb_std_ini=1.0 --ucb_mean_tar=1.0 --ucb_mean_ini=0.0 --filter_goals=none"
EXP_NAME="main_eval"
wandb_project_name="test"
CMD='python training.py  $POINT_MAZE --activate_final --tau_policy=5e-07 --critic_lr=0.0003 --deterministic_eval  --transform_ucb --ensemble_size=6 
--project_name=$wandb_project_name  --goal_log_delay=5 --render_delay=2  --seed=0  
--adaptation_strategy=simple  --relabel_prob_future=0.0 --log_wandb  --group_name=final_eval --exp_name=$EXP_NAME'