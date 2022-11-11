import gym
import ray
import warnings

from tutorials.codes.rllib_train import get_ray_config_and_ray_agent

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


if __name__ == "__main__":
	# ENV_NAME = "CartPole-v1"
	# ENV_NAME = "Taxi-v3"
	# ENV_NAME = "PongDeterministic-v0"
	ENV_NAME = "MountainCarContinuous-v0"

	# ALGORITHM = "DQN"
	ALGORITHM = "PPO"
	# ALGORITHM = "SAC"

	CHECKPOINT_PATH = "/Users/yhhan/ray_results/PPO_MountainCarContinuous-v0_2022-11-07_00-19-04q40ekk4b/checkpoint_000016"

	ray_info = ray.init(log_to_driver=True)

	ray_config, ray_agent = get_ray_config_and_ray_agent(algorithm=ALGORITHM, env_name=ENV_NAME)
	ray_agent.restore(checkpoint_path=CHECKPOINT_PATH)

	env = ray_agent.evaluation_workers.local_worker().env

	for _ in range(3):
		state = env.reset()
		env.render(mode="human")
		done = False
		cumulative_reward = 0

		while not done:
			action = ray_agent.compute_action(state)
			state, reward, done, _ = env.step(action)
			cumulative_reward += reward
			env.render(mode="human")

		print(cumulative_reward)

	ray.shutdown()
