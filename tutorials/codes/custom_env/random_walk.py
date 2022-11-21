import random
import time

import gym

# -------------------------------
# T1 0 1 2 3 4 T2
# -------------------------------
import numpy as np
import torch
from gym.spaces import Discrete, Box
from ray.rllib.env import EnvContext


class RandomWalk(gym.Env):
    def __init__(
        self,
        env_config: EnvContext
    ):
        self.__version__ = "0.0.1"

        self.num_internal_states = env_config["num_internal_states"]

        self.NUM_STATES = env_config["num_internal_states"] + 2

        self.STATES = np.identity(5)

        self.TERMINAL_STATES = [
            np.asarray([-1] * self.num_internal_states),
            np.asarray([-2] * self.num_internal_states)
        ]

        # 모든 가능한 행동
        self.ACTION_LEFT = 0
        self.ACTION_RIGHT = 1
        self.ACTION_SYMBOLS = ["\u2190", "\u2192"]

        # 종료 상태를 제외한 임의의 상태에서 왼쪽 이동 또는 오른쪽 이동
        self.ACTIONS = [
            self.ACTION_LEFT,
            self.ACTION_RIGHT
        ]
        self.NUM_ACTIONS = len(self.ACTIONS)

        self.transition_reward = env_config["transition_reward"]

        self.left_terminal_reward = env_config["left_terminal_reward"]

        self.right_terminal_reward = env_config["right_terminal_reward"]

        self.current_state = None

        # 시작 상태 위치
        self.current_position = int(self.num_internal_states / 2)
        self.START_STATE = self.STATES[self.current_position]

        self.action_space = Discrete(self.NUM_ACTIONS)
        self.observation_space = Box(low=0.0, high=1.0, shape=(self.num_internal_states,))
        self.num_steps = 0

    def reset(self, **kwargs):
        self.num_steps = 0
        self.current_position = int(self.num_internal_states / 2)
        self.current_state = self.STATES[self.current_position]
        return self.current_state

    # take @action in @state
    # @return: (reward, new state)
    def step(self, action):
        self.num_steps += 1
        next_state = self.get_next_state(state=self.current_state, action=action)

        self.current_state = next_state

        reward = self.get_reward(self.current_state, next_state)

        if self.current_position == -1 or self.current_position == self.num_internal_states or self.num_steps == 100:
            done = True
        else:
            done = False

        info = {}

        return next_state, reward, done, info

    def render(self, mode='human'):
        print(self.__str__(), end="\n\n")

    def get_random_action(self):
        return random.choice(self.ACTIONS)

    def moveto(self, state):
        self.current_state = state

    def get_next_state(self, state, action):
        if self.current_position == -1 or self.current_position == self.num_internal_states:
            next_state = state
        else:
            if action == self.ACTION_LEFT:
                self.current_position -= 1
                if self.current_position == -1:
                    next_state = self.TERMINAL_STATES[0]
                else:
                    next_state = self.STATES[self.current_position]

            elif action == self.ACTION_RIGHT:
                self.current_position += 1
                if self.current_position == self.num_internal_states:
                    next_state = self.TERMINAL_STATES[1]
                else:
                    next_state = self.STATES[self.current_position]

            else:
                raise ValueError()

        return next_state

    def get_reward(self, state, next_state):
        if self.current_position == -1:
            reward = self.left_terminal_reward
        elif self.current_position == self.num_internal_states:
            reward = self.right_terminal_reward
        else:
            reward = self.transition_reward

        return reward

    def __str__(self):
        randomwalk_str = ""
        randomwalk_str += " T1 " + " ".join(
            ["{0}".format(i) for i in range(self.num_internal_states)]
        ) + " T2\n"

        if self.current_position == -1:
            blank = " "
        elif self.current_position == self.num_internal_states:
            blank = "  " + "  " * (self.num_internal_states + 1)
        else:
            blank = "    " + "  " * self.current_position

        randomwalk_str += blank + "*"

        return randomwalk_str


if __name__ == "__main__":
    env_config = {
        "num_internal_states": 5,  # 종료 상태를 제외한 내부 상태 개수
        "transition_reward": 0.0,  # 일반적인 상태 전이 보상
        "left_terminal_reward": 0.0,  # 왼쪽 종료 상태로 이동하는 행동 수행 시 받는 보상
        "right_terminal_reward": 1.0  # 오른쪽 종료 상태로 이동하는 행동 수행 시 받는 보상
    }

    env = RandomWalk(env_config)
    env.reset()
    print("reset")
    env.render()

    done = False
    total_steps = 0
    while not done:
        total_steps += 1
        action = env.get_random_action()
        next_state, reward, done, _ = env.step(action)
        print("action: {0}, reward: {1}, done: {2}, total_steps: {3}".format(
            env.ACTION_SYMBOLS[action],
            reward, done, total_steps
        ))
        env.render()

        time.sleep(1)