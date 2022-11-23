import ray
import warnings

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from tutorials.codes.single_agent.rllib_algorithm import ALGORITHM
from tutorials.codes.single_agent.rllib_environment import ENV_NAME
from tutorials.codes.single_agent.rllib_environment import ENV_CONFIG
from tutorials.codes.single_agent.rllib_train import get_ray_config_and_ray_agent


if __name__ == "__main__":
	CHECKPOINT_PATH = "/Users/yhhan/ray_results/PPO_RandomWalk_2022-11-22_21-45-03qyo4z66z/checkpoint_000005"

	ray_info = ray.init(local_mode=True, log_to_driver=True)

	ray_config, ray_agent = get_ray_config_and_ray_agent(algorithm=ALGORITHM, env_name=ENV_NAME, env_config=ENV_CONFIG)
	ray_agent.restore(checkpoint_path=CHECKPOINT_PATH)

	env = ray_agent.evaluation_workers.local_worker().env

	for epsiode in range(3):
		print("[[[ EPISODE: {0} ]]]".format(epsiode))
		state = env.reset()
		env.render(mode="human")
		done = False
		cumulative_reward = 0

		while not done:
			action = ray_agent.compute_single_action(state, explore=False)
			state, reward, done, _ = env.step(action)
			cumulative_reward += reward
			env.render(mode="human")

		print("[EPISODE: {0}] - Cumulative Reward: {1:.2f}".format(
			epsiode, cumulative_reward
		))
		print()

	ray.shutdown()
