POINT_MAZE_SIMPLE="--env_name=high_dimensional_maze --env_file_name=../maze_layouts/rand_2d_10h.pkl --num_timesteps=50000000 --episode_length=100 --num_envs=64 --train_step_multiplier=16 --num_goals=5000 --unroll_length=100 --batch_size=64"
POINT_MAZE_HARD="--env_name=high_dimensional_maze --env_file_name=../maze_layouts/rand_4d_5h.pkl --num_timesteps=50000000 --episode_length=100 --num_envs=64 --train_step_multiplier=16 --num_goals=5000 --unroll_length=100 --batch_size=64"
ANT_MAZE_SIMPLE="--env_name=ant_plain_maze --num_timesteps=50000000 --episode_length=150 --num_envs=32 --train_step_multiplier=12 --num_goals=10000 --unroll_length=150 --batch_size=32"
ANT_MAZE_HARD="--env_name=ant_square_sparse_maze --num_timesteps=100000000 --episode_length=250 --num_envs=64 --train_step_multiplier=16 --num_goals=10000 --unroll_length=250 --batch_size=64"
ARM_SIMPLE="--env_name=arm_push_obstacle_7 --num_timesteps=100000000 --episode_length=250 --num_envs=128 --train_step_multiplier=32 --num_goals=5000 --unroll_length=250 --batch_size=128 --manipulation"
ARM_HARD="--env_name=arm_push_obstacle_2 --num_timesteps=100000000 --episode_length=250 --num_envs=128 --train_step_multiplier=32 --num_goals=5000 --unroll_length=250 --batch_size=128 --manipulation"

SAC="--algo=SAC --goal_selection_strategy=HER --filter_goals=none --relabel_prob_future=0.0"
MAXINFORL="--algo=MaxInfoRL --goal_selection_strategy=HER --filter_goals=none --relabel_prob_future=0.0"
TD3="--algo=TD3 --goal_selection_strategy=HER --filter_goals=none --relabel_prob_future=0.0"
ThompSamp="--algo=ThompTD3 --goal_selection_strategy=HER --filter_goals=none --relabel_prob_future=0.0"
HER="--algo=TD3 --goal_selection_strategy=HER --filter_goals=none --relabel_prob_future=0.7"
UNIFORM="--algo=TD3 --goal_selection_strategy=UNIFORM --filter_goals=MEGA --relabel_prob_future=0.7"
MEGA="--algo=TD3 --goal_selection_strategy=MEGA --filter_goals=MEGA --relabel_prob_future=0.7"
DISCOVER_NO_DIR="--algo=TD3 --goal_selection_strategy=UCB --ucb_std_tar=0.0 --ucb_std_ini=1.0 --ucb_mean_tar=0.0 --ucb_mean_ini=0.0 --filter_goals=none --relabel_prob_future=0.7"
DISCOVER="--algo=TD3 --goal_selection_strategy=UCB --ucb_std_tar=0.0 --ucb_std_ini=1.0 --ucb_mean_tar=1.0 --ucb_mean_ini=0.0 --filter_goals=none --relabel_prob_future=0.7"

EXP_NAME="test"
wandb_project_name="test"
wandb_group_name="test"
ENV=$POINT_MAZE_SIMPLE
ALGO=$DISCOVER

echo "Start Training."
python ../training.py  $ENV $ALGO --project_name=$wandb_project_name --group_name=$wandb_group_name --exp_name=$EXP_NAME \
    --activate_final --tau_policy=5e-07 --critic_lr=0.0003 --deterministic_eval  --transform_ucb --ensemble_size=6 \
    --goal_log_delay=5 --render_delay=2  --seed=0  --adaptation_strategy=simple --log_wandb  
echo "Finished Training."