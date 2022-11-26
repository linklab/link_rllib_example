from gym import spaces
from ray.rllib.policy.policy import PolicySpec


def get_ttt_ray_config_and_ray_agent(algorithm, env_name, env_config, custom_ray_config, num_workers=1):
    if algorithm == "DQN":
        from ray.rllib.algorithms.dqn import DQNConfig
        ray_config = DQNConfig()
    elif algorithm == "PPO":
        from ray.rllib.algorithms.ppo import PPOConfig
        ray_config = PPOConfig()
    elif algorithm == "SAC":
        from ray.rllib.algorithms.sac import SACConfig
        ray_config = SACConfig()
    else:
        raise ValueError()

    ray_config.framework_str = "torch"
    ray_config.env_config = env_config
    ray_config.num_workers = num_workers
    ray_config.evaluation_interval = 1  # 평가를 위한 훈련 간격
    ray_config.evaluation_duration = 5  # 평가를 위한 에피소드 개수

    policies = {
        "policy_O": PolicySpec(
            observation_space=spaces.Box(0.0, 2.0, (12,), float),
            action_space=spaces.Discrete(n=12)
        ),
        "policy_X": PolicySpec(
            observation_space=spaces.Box(0.0, 2.0, (12,), float),
            action_space=spaces.Discrete(n=12)
        )
    }

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id == "O":
            return "policy_O"
        elif agent_id == "X":
            return "policy_X"
        else:
            raise ValueError()

    ray_config.multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
    ray_config.disable_env_checking = True

    if env_config["mode"] == 0:
        ray_config.policies_to_train = ["policy_O"]
    elif env_config["mode"] == 1:
        ray_config.policies_to_train = ["policy_X"]
    elif env_config["mode"] == 2:
        ray_config.policies_to_train = ["policy_O", "policy_X"]
    else:
        raise ValueError()

    ray_config = ray_config.to_dict()
    ray_config.update(custom_ray_config)

    if algorithm == "DQN":
        from ray.rllib.algorithms.dqn import DQN
        ray_agent = DQN(config=ray_config, env=env_name)
    elif algorithm == "PPO":
        from ray.rllib.algorithms.ppo import PPO
        ray_agent = PPO(config=ray_config, env=env_name)
    elif algorithm == "SAC":
        from ray.rllib.algorithms.sac import SAC
        ray_agent = SAC(config=ray_config, env=env_name)
    else:
        raise ValueError()

    return ray_config, ray_agent


def print_ttt_iter_result(
        iter_result, num_optimizations_policy_O, num_optimizations_policy_X,
        evaluation_episode_reward_policy_O_mean, evaluation_episode_reward_policy_X_mean
):
    prefix = "{0:>2}|episodes: {1:>3}|timesteps: {2:>7,}|time: {3:>6.2f} sec.".format(
        iter_result["training_iteration"], iter_result["episodes_total"], iter_result["timesteps_total"], iter_result["time_total_s"]
    )

    episode_reward_policy_O = "epi_reward_mean: {0:>6.2f}, opts: {1:>6,}".format(
        iter_result["sampler_results"]["policy_reward_mean"]["policy_O"], int(num_optimizations_policy_O)
    )

    episode_reward_policy_X = "epi_reward_mean: {0:>6.2f}, opts: {1:>6,}".format(
        iter_result["sampler_results"]["policy_reward_mean"]["policy_X"], int(num_optimizations_policy_X)
    )

    evaluation_episode_reward_policy_O = "eval_epi_reward_mean: {0:>6.2f}".format(
        evaluation_episode_reward_policy_O_mean
    )

    evaluation_episode_reward_policy_X = "eval_epi_reward_mean: {0:>6.2f}".format(
        evaluation_episode_reward_policy_X_mean
    )

    loss_policy_O = "avg. loss: N/A"
    if "policy_O" in iter_result["info"]["learner"]:
        if 'mean_td_error' in iter_result["info"]["learner"]["policy_O"]:
            loss_policy_O = "avg. loss: {0:>6.3f}".format(
                iter_result["info"]["learner"]["policy_O"]["mean_td_error"]
            )
        else:
            loss_policy_O = "avg. loss: {0:>6.3f}".format(
                iter_result["info"]["learner"]["policy_O"]["learner_stats"]["total_loss"]
            )

    loss_policy_X = "avg. loss: N/A"
    if "policy_X" in iter_result["info"]["learner"]:
        if 'mean_td_error' in iter_result["info"]["learner"]["policy_X"]:
            loss_policy_X = "avg. loss: {0:>6.3f}".format(
                iter_result["info"]["learner"]["policy_X"]["mean_td_error"]
            )
        else:
            loss_policy_X = "avg. loss: {0:>6.3f}".format(
                iter_result["info"]["learner"]["policy_X"]["learner_stats"]["total_loss"]
            )

    print("[{0}] [policy_O: {1}, {2}, {3}] [policy_X:{4}, {5}, {6}]".format(
        prefix,
        episode_reward_policy_O, evaluation_episode_reward_policy_O, loss_policy_O,
        episode_reward_policy_X, evaluation_episode_reward_policy_X, loss_policy_X,
    ))


def log_ttt_wandb(
        wandb, iter_result, num_optimizations_policy_O, num_optimizations_policy_X,
        evaluation_episode_reward_policy_O_mean, evaluation_episode_reward_policy_X_mean
):
    log_dict = {
        "train": iter_result["training_iteration"],
        "episodes": iter_result["episodes_total"],
        "timesteps": iter_result["timesteps_total"],
        "policy_O/optimizations": num_optimizations_policy_O,
        "policy_O/episode_reward_mean": iter_result["sampler_results"]["policy_reward_mean"]["policy_O"],
        "policy_O/evaluation_episode_reward_mean": evaluation_episode_reward_policy_O_mean,
        "policy_X/optimizations": num_optimizations_policy_X,
        "policy_X/episode_reward_mean": iter_result["sampler_results"]["policy_reward_mean"]["policy_X"],
        "policy_X/evaluation_episode_reward_mean": evaluation_episode_reward_policy_X_mean,
    }

    if "policy_O" in iter_result["info"]["learner"]:
        if 'mean_td_error' in iter_result["info"]["learner"]["policy_O"]:
            log_dict["policy_O/loss"] = iter_result["info"]["learner"]["policy_O"]["mean_td_error"]
        else:
            log_dict["policy_O/loss"] = iter_result["info"]["learner"]["policy_O"]["learner_stats"]["total_loss"]

    if "policy_X" in iter_result["info"]["learner"]:
        if 'mean_td_error' in iter_result["info"]["learner"]["policy_X"]:
            log_dict["policy_X/loss"] = iter_result["info"]["learner"]["policy_X"]["mean_td_error"]
        else:
            log_dict["policy_X/loss"] = iter_result["info"]["learner"]["policy_X"]["learner_stats"]["total_loss"]

    wandb.log(log_dict)
