import warnings
from pprint import pprint
import numpy as np

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import ray
import wandb
from datetime import datetime

from tutorials.codes.single_agent.rllib_utils import get_ray_config_and_ray_agent, print_iter_result, log_wandb
from tutorials.codes.single_agent.rllib_algorithm import ALGORITHM
from tutorials.codes.single_agent.rllib_environment import (
	ENV_NAME, CUSTOM_RAY_CONFIG, ENV_CONFIG, MAX_TRAIN_ITERATIONS, EPISODE_REWARD_AVG_SOLVED, NUM_EPISODES_EVALUATION
)

import gym


class RAY_RL:
	def __init__(
			self, env_name, algorithm, ray_config, ray_agent, max_train_iterations, episode_reward_mean_solved, use_wandb
	):
		self.env_name = env_name
		self.algorithm = algorithm
		self.ray_config = ray_config
		self.ray_agent = ray_agent

		self.max_train_iterations = max_train_iterations
		self.episode_reward_mean_solved = episode_reward_mean_solved
		self.use_wandb = use_wandb

		self.current_time = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

		if self.use_wandb:
			self.wandb = wandb.init(
				project="{0}_{1}".format(self.algorithm, self.env_name),
				name=self.current_time,
				config=self.ray_config
			)

		self.test_env = gym.make(self.env_name)

	def train_loop(self):
		num_optimizations = 0

		for num_train in range(self.max_train_iterations):
			try:
				iter_result = self.ray_agent.train()

				# from pprint import pprint
				# pprint(iter_result)
				if "num_agent_steps_trained" in iter_result["info"]:
					num_optimizations += iter_result["info"]["num_agent_steps_trained"]
				else:
					if "default_policy" in iter_result["info"]["learner"]:
						num_optimizations += iter_result["info"]["learner"]["default_policy"]["num_agent_steps_trained"]
					else:
						num_optimizations += 0

				(
					evaluation_episode_reward_min,
					evaluation_episode_reward_max,
					evaluation_episode_reward_mean,
					evaluation_episode_steps_list,
					evaluation_episode_steps_mean
				) = self.evaluate()

				print_iter_result(
					iter_result,
					num_optimizations,
					evaluation_episode_reward_mean,
					evaluation_episode_steps_mean,
					NUM_EPISODES_EVALUATION
				)

				if self.use_wandb:
					log_wandb(
						self.wandb,
						iter_result,
						num_optimizations,
						evaluation_episode_reward_mean, evaluation_episode_reward_min, evaluation_episode_reward_max,
						evaluation_episode_steps_mean
					)

				if evaluation_episode_reward_mean >= self.episode_reward_mean_solved and num_optimizations > 50_000:
					checkpoint_path = ray_agent.save()
					print("*** Solved with Evaluation Episodes Reward Mean: {0:>6.2f} ({1} Evaluation Episodes).".format(
						iter_result["evaluation"]["episode_reward_mean"],
						iter_result["evaluation"]["episodes_this_iter"]
					))
					print("*** Checkpoint at {0}".format(checkpoint_path))
					break
			except ValueError as e:
				print(e, "--> ValueError")

	def evaluate(self):
		evaluation_episode_reward_lst = []
		evaluation_episode_steps_lst = []

		for i in range(NUM_EPISODES_EVALUATION):
			episode_reward = 0.0
			episode_steps = 0

			observation = self.test_env.reset()

			done = False

			while not done:
				action = self.ray_agent.compute_single_action(observation, explore=False)

				next_observation, reward, done, info = self.test_env.step(action)

				episode_reward += reward
				observation = next_observation
				episode_steps += 1

			evaluation_episode_reward_lst.append(episode_reward)
			evaluation_episode_steps_lst.append(episode_steps)

		return (
			min(evaluation_episode_reward_lst),
			max(evaluation_episode_reward_lst),
			np.average(evaluation_episode_reward_lst),
			evaluation_episode_steps_lst,
			np.average(evaluation_episode_steps_lst)
		)


if __name__ == "__main__":
	print("RAY VERSION: {0}".format(ray.__version__))
	print("GYM VERSION: {0}".format(gym.__version__))

	ray_info = ray.init(local_mode=True)

	ray_config, ray_agent = get_ray_config_and_ray_agent(
		algorithm=ALGORITHM,
		env_name=ENV_NAME,
		env_config=ENV_CONFIG,
		custom_ray_config=CUSTOM_RAY_CONFIG,
		num_workers=1
	)

	pprint(ray_config)

	print(ray_agent.get_policy().model)
	print("OBSERVATION SPACE: {0}".format(str(ray_agent.get_policy().observation_space)))
	print("ACTION SPACE: {0}".format(str(ray_agent.get_policy().action_space)))

	ray_rl = RAY_RL(
		env_name=ENV_NAME, algorithm=ALGORITHM, ray_config=ray_config, ray_agent=ray_agent,
		max_train_iterations=MAX_TRAIN_ITERATIONS, episode_reward_mean_solved=EPISODE_REWARD_AVG_SOLVED,
		use_wandb=False
	)

	ray_rl.train_loop()

	ray.shutdown()
