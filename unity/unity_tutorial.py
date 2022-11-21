import platform
import os
import torch
from mlagents_envs.environment import UnityEnvironment

game_name = "Kart"
os_name = platform.system()

if os_name == 'Windows':
    env_name = f"../unity_env/{game_name}_{os_name}/{game_name}"
elif os_name == 'Darwin':
    env_name = f"../unity_env/{game_name}_{os_name}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    env = UnityEnvironment(file_name=env_name)

    env.reset()
    behavior_name = list(env.behavior_specs.keys())[0]

    print("behavior name: ", behavior_name)

    spec = env.behavior_specs[behavior_name]

    for episode in range(10):
        env.reset()

        decision_steps, terminal_steps = env.get_steps(behavior_name)

        tracked_agent = -1
        done = False
        episode_rewards = 0

        while not done:
            if tracked_agent == -1 and len(decision_steps) >= 1:
                tracked_agent = decision_steps.agent_id[0]

            action = spec.action_spec.random_action(len(decision_steps))

            env.set_actions(behavior_name, action)

            env.step()

            decision_steps, terminal_steps = env.get_steps(behavior_name)

            if tracked_agent in decision_steps:
                episode_rewards += decision_steps[tracked_agent].reward
            if tracked_agent in terminal_steps:
                episode_rewards += terminal_steps[tracked_agent].reward
                done = True

        print("episode: {0}, episode reward: {1}".format(episode, episode_rewards))

    env.close()
