import warnings

import numpy as np

from tutorials.codes.custom_env.tic_tac_toe_343 import TicTacToe343

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import ray
import wandb
from datetime import datetime

from tutorials.codes.multi_agents.rllib_ttt_utils import (
	get_ttt_ray_config_and_ray_agent, print_ttt_iter_result, log_ttt_wandb
)
from tutorials.codes.multi_agents.rllib_ttt_algorithm import ALGORITHM
from tutorials.codes.multi_agents.rllib_ttt_environment import (
	ENV_NAME, CUSTOM_RAY_CONFIG, ENV_CONFIG, MAX_TRAIN_ITERATIONS, EPISODE_REWARD_AVG_SOLVED, NUM_EPISODES_EVALUATION
)

import gym


class RAY_RL:
	def __init__(
			self, env_name, algorithm, ray_config, ray_agent, max_train_iterations, episode_reward_avg_solved, use_wandb
	):
		self.env_name = env_name
		self.algorithm = algorithm
		self.ray_config = ray_config
		self.ray_agent = ray_agent

		self.max_train_iterations = max_train_iterations
		self.episode_reward_avg_solved = episode_reward_avg_solved
		self.use_wandb = use_wandb

		self.current_time = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

		if self.use_wandb:
			self.wandb = wandb.init(
				project="{0}_{1}".format(self.algorithm, self.env_name),
				name=self.current_time,
				config=self.ray_config
			)

		self.test_env = TicTacToe343(env_config=ENV_CONFIG)

	def train_loop(self):
		num_optimizations_policy_O = 0
		num_optimizations_policy_X = 0

		for num_train in range(self.max_train_iterations):
			iter_result = self.ray_agent.train()

			# from pprint import pprint
			# pprint(iter_result)
			if "policy_O" in iter_result["info"]["learner"]:
				num_optimizations_policy_O += iter_result["info"]["learner"]["policy_O"]["num_agent_steps_trained"]

			if "policy_X" in iter_result["info"]["learner"]:
				num_optimizations_policy_X += iter_result["info"]["learner"]["policy_X"]["num_agent_steps_trained"]

			evaluation_episode_reward_policy_O_mean, evaluation_episode_reward_policy_X_mean = self.evaluate()

			print_ttt_iter_result(
				iter_result,
				num_optimizations_policy_O, num_optimizations_policy_X,
				evaluation_episode_reward_policy_O_mean, evaluation_episode_reward_policy_X_mean
			)

			if self.use_wandb:
				log_ttt_wandb(
					self.wandb, iter_result, num_optimizations_policy_O, num_optimizations_policy_X,
					evaluation_episode_reward_policy_O_mean, evaluation_episode_reward_policy_X_mean
				)

			if ENV_CONFIG["mode"] in [0, 1]:
				if ENV_CONFIG["mode"] == 0:
					episode_reward_mean = iter_result["evaluation"]["policy_reward_mean"]["policy_O"]
				else:
					episode_reward_mean = iter_result["evaluation"]["policy_reward_mean"]["policy_X"]

				if episode_reward_mean >= self.episode_reward_avg_solved:
					checkpoint_path = ray_agent.save()
					print("*** Solved with Evaluation Episodes Reward Mean: {0:>6.2f} ({1} Evaluation Episodes).".format(
						episode_reward_mean, iter_result["evaluation"]["episodes_this_iter"]
					))
					print("*** Checkpoint at {0}".format(checkpoint_path))
					break

		if ENV_CONFIG["mode"] == 2:
			checkpoint_path = ray_agent.save()
			print("*** Final Checkpoint at {0}".format(checkpoint_path))

	def evaluate(self):
		evaluation_episode_reward_policy_O_lst = []
		evaluation_episode_reward_policy_X_lst = []

		for i in range(NUM_EPISODES_EVALUATION):
			observations = self.test_env.reset()
			dones = dict()
			dones["__all__"] = False

			cumulative_reward = {
				'O': 0.0,
				'X': 0.0
			}

			policy_id = 'policy_O'

			while not dones["__all__"]:
				actions = ray_agent.compute_actions(observations, policy_id=policy_id, explore=False)

				next_observations, rewards, dones, infos = self.test_env.step(actions)

				print(
					"Obs.: {0}, Action: {1}, Next Obs.: {2}, Reward: {3}, Done: {4}, Info: {5}".format(
						observations, actions, next_observations, rewards, dones, infos
					))

				if 'O' in rewards:
					cumulative_reward['O'] += rewards['O']

				if 'X' in rewards:
					cumulative_reward['X'] += rewards['X']

				policy_id = "policy_X" if policy_id == "policy_O" else "policy_O"
				print(policy_id, "#####################")

				observations = next_observations

			evaluation_episode_reward_policy_O_lst.append(cumulative_reward['O'])
			evaluation_episode_reward_policy_X_lst.append(cumulative_reward['X'])

		return (
			np.average(evaluation_episode_reward_policy_O_lst),
			np.average(evaluation_episode_reward_policy_X_lst)
		)


if __name__ == "__main__":
	print("RAY VERSION: {0}".format(ray.__version__))
	print("GYM VERSION: {0}".format(gym.__version__))

	ray_info = ray.init(local_mode=True)

	ray_config, ray_agent = get_ttt_ray_config_and_ray_agent(
		algorithm=ALGORITHM,
		env_name=ENV_NAME,
		env_config=ENV_CONFIG,
		custom_ray_config=CUSTOM_RAY_CONFIG,
		num_workers=0
	)

	print("#" * 128)
	print(ray_agent.get_policy(policy_id="policy_O").model)
	print("OBSERVATION SPACE: {0}".format(str(ray_agent.get_policy(policy_id="policy_O").observation_space)))
	print("ACTION SPACE: {0}".format(str(ray_agent.get_policy(policy_id="policy_O").action_space)))

	print("#" * 128)
	print(ray_agent.get_policy(policy_id="policy_X").model)
	print("OBSERVATION SPACE: {0}".format(str(ray_agent.get_policy(policy_id="policy_X").observation_space)))
	print("ACTION SPACE: {0}".format(str(ray_agent.get_policy(policy_id="policy_X").action_space)))

	ray_rl = RAY_RL(
		env_name=ENV_NAME, algorithm=ALGORITHM, ray_config=ray_config, ray_agent=ray_agent,
		max_train_iterations=MAX_TRAIN_ITERATIONS, episode_reward_avg_solved=EPISODE_REWARD_AVG_SOLVED,
		use_wandb=False
	)

	ray_rl.train_loop()

	ray.shutdown()
