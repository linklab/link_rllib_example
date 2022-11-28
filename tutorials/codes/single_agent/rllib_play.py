import gym
import ray
import warnings

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from tutorials.codes.single_agent.rllib_algorithm import ALGORITHM
from tutorials.codes.single_agent.rllib_environment import ENV_NAME, ENV_CONFIG

from tutorials.codes.single_agent.rllib_train import get_ray_config_and_ray_agent


if __name__ == "__main__":

	ray_info = ray.init(local_mode=True, log_to_driver=True)

	ray_config, ray_agent = get_ray_config_and_ray_agent(
		algorithm=ALGORITHM,
		env_name=ENV_NAME,
		env_config=ENV_CONFIG
	)
	ray_agent.restore(
		checkpoint_path="/Users/yhhan/ray_results/PPO_RandomWalk_2022-11-22_21-45-03qyo4z66z/checkpoint_000005"
	)

	env = gym.make(ENV_NAME)

	for epsiode in range(3):
		print("[[[ EPISODE: {0} ]]]".format(epsiode))
		observation = env.reset()
		env.render(mode="human")

		episode_reward = 0.0
		episode_steps = 0

		done = False

		while not done:
			action = ray_agent.compute_single_action(observation, explore=False)
			next_observation, reward, done, _ = env.step(action)

			env.render(mode="human")

			episode_reward += reward
			observation = next_observation
			episode_steps += 1

		print("[EPISODE: {0}] - Episode Reward: {1:.2f} (Steps: {2})".format(
			epsiode, episode_reward, episode_steps
		))
		print()

	ray.shutdown()
