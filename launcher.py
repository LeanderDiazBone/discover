import argparse
import sys, os
import copy
import training
import training_td3
import training_sac
#parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#sys.path.append(parent_dir)
from launcher_utils import generate_base_command, generate_run_commands

ENV_CONFIG = {
    "ant_plain_maze": {"num_timesteps": 50*10**6, "episode_length": 150, "num_envs": 32, "train_step_multiplier": 12, "num_goals": 10000},
    "ant_square_sparse_maze": {"num_timesteps": 100*10**6, "episode_length": 250, "num_envs": 64, "train_step_multiplier": 16, "num_goals": 10000}, #50000000
    "ant_long_hor_maze": {"num_timesteps": 200*10**6, "episode_length": 500, "num_envs": 128, "train_step_multiplier": 32, "num_goals": 15000},
    "high_dimensional_maze": {"num_timesteps": 50*10**6, "episode_length": 100, "num_envs": 64, "train_step_multiplier": 16, "num_goals": 5000},
    "arm_reach": {"num_timesteps":  100*10**6, "episode_length": 100, "num_envs": 128, "train_step_multiplier": 32, "num_goals": 10000},
    "arm_push_easy": {"num_timesteps":  50*10**6, "episode_length": 100, "num_envs": 128, "train_step_multiplier": 32, "num_goals": 10000},
    "arm_push_hard": {"num_timesteps":  100*10**6, "episode_length": 100, "num_envs": 128, "train_step_multiplier": 32, "num_goals": 10000},
    #"arm_push_obstacle": {"num_timesteps": 100*10**6, "episode_length": 100, "num_envs": 64, "train_step_multiplier": 16, "num_goals": 5000},
    "arm_push_obstacle_0": {"num_timesteps": 100*10**6, "episode_length": 250, "num_envs": 128, "train_step_multiplier": 32, "num_goals": 5000},
    "arm_push_obstacle_1": {"num_timesteps": 100*10**6, "episode_length": 250, "num_envs": 128, "train_step_multiplier": 32, "num_goals": 5000},
    "arm_push_obstacle_2": {"num_timesteps": 100*10**6, "episode_length": 250, "num_envs": 128, "train_step_multiplier": 32, "num_goals": 5000},
    "arm_push_obstacle_3": {"num_timesteps": 100*10**6, "episode_length": 250, "num_envs": 128, "train_step_multiplier": 32, "num_goals": 5000},
    "arm_push_obstacle_4": {"num_timesteps": 50*10**6, "episode_length": 250, "num_envs": 128, "train_step_multiplier": 32, "num_goals": 5000},
    "arm_push_obstacle_5": {"num_timesteps": 100*10**6, "episode_length": 250, "num_envs": 128, "train_step_multiplier": 32, "num_goals": 5000},
    "arm_push_obstacle_6": {"num_timesteps": 100*10**6, "episode_length": 250, "num_envs": 128, "train_step_multiplier": 32, "num_goals": 5000},
    "arm_push_obstacle_7": {"num_timesteps": 100*10**6, "episode_length": 250, "num_envs": 128, "train_step_multiplier": 32, "num_goals": 5000},
    "arm_push_obstacle_8": {"num_timesteps": 100*10**6, "episode_length": 250, "num_envs": 128, "train_step_multiplier": 32, "num_goals": 5000},
    "arm_push_obstacle_9": {"num_timesteps": 100*10**6, "episode_length": 250, "num_envs": 128, "train_step_multiplier": 32, "num_goals": 5000},
    "simple_square_sparse_maze": {"num_timesteps": 50*10**6, "episode_length": 250, "num_envs": 64, "train_step_multiplier": 16, "num_goals": 5000}, 
    "simple_plain_maze": {"num_timesteps": 25*10**6, "episode_length": 100, "num_envs": 64, "train_step_multiplier": 16, "num_goals": 5000}, 
    "ant_big_sparse_maze": {"num_timesteps":300000000, "episode_length": 2500}, 
    "ant_large_plain_maze": {"num_timesteps": 25000000, "episode_length": 250},
    "ant_huge_maze": {"num_timesteps": 500000000, "episode_length": 1000},
    "ant_big_square_sparse_maze": {"num_timesteps": 50000000, "episode_length": 250},
    "simple_long_hor_maze": {"num_timesteps": 100000000, "episode_length": 2000},
    "ant_corner_maze": {"num_timesteps": 300000000, "episode_length": 2000},
    "simple_large_plain_maze": {"num_timesteps": 100000000, "episode_length": 2000}, 
    "ant": {"num_timesteps":60000000, "episode_length": 750},     
    "ant_big_maze": {"num_timesteps": 150*10**6, "episode_length": 500, "num_envs": 128, "train_step_multiplier": 32, "num_goals": 50000},
}

applicable_configs = {
    "group_name": ["final_eval"],#final_eval
    "env_name": ["high_dimensional_maze"],      #arm_push_obstacle_2"arm_push_hard", "ant_plain_maze","ant_square_sparse_maze", "ant_square_sparse_maze", "ant_square_sparse_maze", "high_dimensional_maze", ant_square_sparse_maze,  "ant_long_hor_maze"ant_plain_maze "ant_square_sparse_maze", "ant_plain_maze", "ant_plain_maze", , "arm_push_hard", high_dimensional_maze
    "env_file_name": ["rand_4d_5h"],    #"rand_3d_7h", "rand_5d_4h"rand_2d_10h", "rand_3d_8h", "rand_4d_6h, 2d_15h", "rand_3d_8h, "2d_15h", rand_4d_5h, "rand_2d_10h", "rand_3d_8h", "rand_4d_5h", rand_2d_10h", "rand_4d_5h
    "exp_name_add": [""],
    "algo": ["TD3", "SAC", "MaxInfoSAC"], # ,  "ThompTD3", SAC, , , "TD3", "SAC", , ThompTD3", "SAC", "MaxInfoSAC
    "relabel_prob_future": [0.0],
    "goal_selection_strategy": ["HER"], #,"ORACLE", , "MEGA", , "UCB"
    "adaptation_strategy": ["simple"],#simple 
    "adaptation_rate": [100],
    #"filter_goals": ["MEGA"],
    "ucb_mean_ini": [0.0],
    "ucb_mean_tar": [0.0],
    "ucb_std_ini":  [1.0],
    "ucb_std_tar":  [0.0],
    "mega_cutoff_step_size": [1],
    "seed": [0,1,2,3,4,5,6,7,8,9],  #,6,7,8,9
    #"critic_prior_file_name": ["mega_square_maze"], # mega_plain_maze,mega_square_maze
    "cur_alpha": [0.0],#0.0,, 1.0, , 0.25,, 5, 10
    "obs_size_prior": [6],
    "action_size_prior": [2],
    "eval_target_policy": [False],
    "render_delay": [2],
    "goal_log_delay": [2],
    "grid_size": [0], # 15
    "goal_achievement_target": [0.5],
    "dir_var": [0.0],
    "save_models": [False],#True
}
"""
applicable_configs = {
    "group_name": ["final_eval"],#final_eval
    "env_name": ["arm_push_obstacle_2"],      #"arm_push_hard", "ant_plain_maze","ant_square_sparse_maze", "ant_square_sparse_maze", "ant_square_sparse_maze", "high_dimensional_maze", ant_square_sparse_maze,  "ant_long_hor_maze"ant_plain_maze "ant_square_sparse_maze", "ant_plain_maze", "ant_plain_maze", , "arm_push_hard", high_dimensional_maze
    "env_file_name": [""],    #"rand_3d_7h", "rand_5d_4h"rand_2d_10h", "rand_3d_8h", "rand_4d_6h, 2d_15h", "rand_3d_8h, "2d_15h", rand_4d_5h, "rand_2d_10h", "rand_3d_8h", "rand_4d_5h", rand_2d_10h", "rand_4d_5h
    "exp_name_add": [""],
    "algo": ["TD3"], # , "ThompTD3" SAC, , "TD3", "TD3", "SAC",, "SAC", "MaxInfoSAC"
    "relabel_prob_future": [0.7],
    "goal_selection_strategy": ["HER", "UNIFORM","UCB"], #, "MEGA""MEGA",, #,"ORACLE", , "MEGA", , "UCB", "HER", "UNIFORM", 
    "adaptation_strategy": ["simple"],#simple 
    "adaptation_rate": [100],
    #"filter_goals": ["MEGA"],
    "ucb_mean_ini": [0.0],
    "ucb_mean_tar": [0.0],
    "ucb_std_ini":  [1.0],
    "ucb_std_tar":  [0.0],
    "mega_cutoff_step_size": [1],
    "seed": [0,1,2,3,4,5,6,7,8,9],  #,6,7,8,9
    #"critic_prior_file_name": ["mega_square_maze"], # mega_plain_maze,mega_square_maze
    "cur_alpha": [0.0],#0.0,, 1.0, , 0.25,, 5, 10
    "obs_size_prior": [6],
    "action_size_prior": [2],
    "eval_target_policy": [False],
    "render_delay": [2],
    "goal_log_delay": [2],
    "grid_size": [0], # 15
    "goal_achievement_target": [0.5],
    "dir_var": [0.0],
    "save_models": [False],#True
}"""
def adjust_applicable_configs(applicable_configs):
    applicable_configs["project_name"] =  ["information-directed-gcrl-jax"]
    
    applicable_configs["ensemble_size"] = [6] # Needs to be even
    applicable_configs["use_her"] = [True]
    applicable_configs["discounting"] = [0.99]
    applicable_configs["action_repeat"] = [1]
    applicable_configs["min_replay_size"] = [1000]
    applicable_configs["max_replay_size"] = [25000]
    applicable_configs["num_evals"] = [250] # 100
    applicable_configs["negative_rewards"] = [False]
    applicable_configs["use_metric"] = [False]
    applicable_configs["transform_ucb"] =  [True]#True
    applicable_configs["standardize_ucb"] =  [False]
    applicable_configs["target_computation"] = ["min_random"] # "single"
    applicable_configs["continue_strategy"] = ["uniform"]
    applicable_configs["manipulation"] = ["arm" in applicable_configs["env_name"][0]]
    
    applicable_configs["log_wandb"] = [True]
    applicable_configs["deterministic_eval"] = [True]
    
    applicable_configs["critic_lr"] = [3e-4]
    applicable_configs["tau"] = [0.005]
    applicable_configs["tau_policy"] = [0.0000005]
    applicable_configs["policy_delay"] = [2]
    applicable_configs["activate_final"] = [True] # False
    #applicable_configs["batch_size"] = [64] # 512
    applicable_configs["relabel_prob_uniform"] = [0.0]
    applicable_configs["relabel_prob_geometric"] = [0.0]

    return applicable_configs

#applicable_configs = applicable_configs_ant_square_maze
applicable_configs = adjust_applicable_configs(applicable_configs)
#applicable_configs = applicable_configs_arm_push_easy
def generate_flag_list_recursively(keys):
    flag_list = []
    if len(keys) == 1:
        for el in applicable_configs[keys[0]]:
            flags = {keys[0]: el}
            flag_list.append(flags)
    else:
        flag_list_rec = generate_flag_list_recursively(keys[1:])
        for el in applicable_configs[keys[0]]:
            for flag_dic_r in flag_list_rec:
                flag_dir_r_copy = copy.deepcopy(flag_dic_r)
                flag_dir_r_copy[keys[0]] =  el
                flag_list.append(flag_dir_r_copy)
    return flag_list

def make_experiment_name(cmd, mode):
    if cmd["algo"] == 'TD3':
        cmd["exploration_noise"] = 0.4
    else:
        cmd["exploration_noise"] = 0.0
    if cmd["goal_selection_strategy"] in ["UNIFORM", "MEGA"]:
        cmd["filter_goals"] = "MEGA"
    else:
        cmd["filter_goals"] = "none"
    #cmd["filter_goals"] = "none"
    for key in ENV_CONFIG[cmd["env_name"]].keys():
        cmd[key] = ENV_CONFIG[cmd["env_name"]][key]
    cmd["exp_name_add"] = cmd["exp_name_add"]+"_"+cmd["group_name"]
    #cmd["train_step_multiplier"] = cmd["train_step_multiplier"]
    env_add = ""
    if "critic_prior_file_name" in cmd.keys():
        if mode == "local":
            cmd["critic_prior_file_name"] = "~/Developer/information-directed-gcrl-jax/prior_critic/" + cmd["critic_prior_file_name"] + ".pkl" 
        else:
            cmd["critic_prior_file_name"] = "~/information-directed-gcrl-jax/prior_critic/" + cmd["critic_prior_file_name"] + ".pkl" 
    if cmd["env_name"] == "high_dimensional_maze":
        env_add = "_" + cmd["env_file_name"] 
        if mode == "local":
            cmd["env_file_name"] = "~/Developer/information-directed-gcrl-jax/maze_layouts/" + cmd["env_file_name"] + ".pkl" 
        else:
            cmd["env_file_name"] = "~/information-directed-gcrl-jax/maze_layouts/" + cmd["env_file_name"] + ".pkl" 
    
    testing = False
    if testing:
        cmd["num_timesteps"] = 100000
        cmd["episode_length"] = 20
        cmd["num_envs"] = 1  
        cmd["log_wandb"] = False 
        cmd["train_step_multiplier"] = 1 
    algo = cmd["algo"]
    cmd["unroll_length"] = cmd["episode_length"]
    #cmd["num_goals"] = cmd["episode_length"] * int(cmd["num_envs"]/2)
    cmd["batch_size"] = cmd["num_envs"]
    env_name = cmd["env_name"]
    exp_name_add = cmd["exp_name_add"]# + "_" + cmd["group_name"]
    goal_selection = cmd["goal_selection_strategy"]
    filter_goals = cmd["filter_goals"]
    mega_cutoff_step_size = cmd["mega_cutoff_step_size"]
    if filter_goals == "MEGA":
        filter_goals += f"_{mega_cutoff_step_size}"
    ucb_mean_ini = cmd["ucb_mean_ini"]
    ucb_mean_tar = cmd["ucb_mean_tar"]
    ucb_std_ini = cmd["ucb_std_ini"]
    ucb_std_tar = cmd["ucb_std_tar"]
    critic_lr = cmd["critic_lr"]
    trainstep_multiplier = cmd["train_step_multiplier"]
    num_envs = cmd["num_envs"]
    continue_strategy = cmd["continue_strategy"]
    relabel_prob_future = cmd["relabel_prob_future"]
    relabel_prob_uniform = cmd["relabel_prob_uniform"]
    relabel_prob_geometric = cmd["relabel_prob_geometric"]
    adaptation_rate = cmd["adaptation_rate"]
    goal_achievement_target = cmd["goal_achievement_target"]
    cur_alpha = cmd["cur_alpha"]
    if adaptation_rate:
        adaptation_strategy = cmd["adaptation_strategy"] + "_" + str(adaptation_rate) + "_" + str(goal_achievement_target)
    else:
        adaptation_strategy = cmd["adaptation_strategy"]
    tau = cmd["tau"]
    policy_delay = cmd["policy_delay"]
    batch_size = cmd["batch_size"]
    target_computation = cmd["target_computation"]
    dir_var = cmd["dir_var"]
    if "critic_prior_file_name" in cmd.keys():
        prior = "_prior"
    elif cmd["use_metric"]:
        prior = "_metric"
    else:
        prior = ""
    learning_params = f"{critic_lr}_{tau}_{policy_delay}_{batch_size}_relabel_{relabel_prob_future}_{relabel_prob_uniform}_{relabel_prob_geometric}_{target_computation}{prior}_{trainstep_multiplier}_{num_envs}_{continue_strategy}_{adaptation_strategy}_{cur_alpha}"
    if cmd["activate_final"]:
        last_layer = "_softplus"
    else:
        last_layer = ""
    if goal_selection == "UCB" or goal_selection == "ORACLE":
        if dir_var > 0:
            cmd["exp_name"] = f"{algo}_{goal_selection}_{dir_var}_{ucb_mean_ini}_{ucb_mean_tar}_{ucb_std_ini}_{ucb_std_tar}_{env_name}{env_add}_filter_{filter_goals}{last_layer}_{mode}_{learning_params}{exp_name_add}"
        else:
            cmd["exp_name"] = f"{algo}_{goal_selection}_{ucb_mean_ini}_{ucb_mean_tar}_{ucb_std_ini}_{ucb_std_tar}_{env_name}{env_add}_filter_{filter_goals}{last_layer}_{mode}_{learning_params}{exp_name_add}"
    else:
        cmd["exp_name"] = f"{algo}_{goal_selection}_{env_name}{env_add}_filter_{filter_goals}{last_layer}_{mode}_{learning_params}{exp_name_add}"
    del cmd["exp_name_add"]
    return cmd        

def main(args):
    command_list = []
    flag_list = generate_flag_list_recursively(list(applicable_configs.keys()))
    flag_list_2 = []
    if args.local:
        mode = "local"
        promt = False
    else:
        mode = "euler"
        promt = True
    for cmd in flag_list:
        cmd = make_experiment_name(cmd, mode)
        flag_list_2.append(cmd)
    for flags in flag_list_2:
        cmd = generate_base_command(training_td3, flags=flags)
        command_list.append(cmd)
    print(command_list)
    
    generate_run_commands(
        command_list,
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        mode=mode,
        num_hours=args.num_hours,
        promt=promt,
        mem=16000,
        #gpu_type=args.gpu_type
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cpus", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-hours", type=int, default=50)
    parser.add_argument("--gpu-type", type=str, default="rtx_4090")
    parser.add_argument("--local", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    main(args)
