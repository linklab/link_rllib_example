import time

import gym
import numpy as np
from gym import spaces
from ray.rllib import MultiAgentEnv
from ray.rllib.env import EnvContext

from tutorials.codes.multi_agents.agents.a_dummy_agent import Dummy_Agents

PLAYER_TO_SYMBOL = [' ', 'O', 'X']
PLAYER_1_INT = 1
PLAYER_2_INT = 2


#########################################################
##  (0,0) -> 0, (0,1) ->  1, (0,2) ->  2, (0,3) ->  3  ##
##  (1,0) -> 4, (1,1) ->  5, (1,2) ->  6, (1,3) ->  7  ##
##  (2,0) -> 8, (2,1) ->  9, (2,2) -> 10, (2,3) -> 11  ##
#########################################################
def position_to_action_idx(row_idx, col_idx):
    return 3 * row_idx + col_idx


#########################################################
##  0 -> (0,0),  1 -> (0,1),  2 -> (0,2),  3 -> (0,3)  ##
##  4 -> (1,0),  5 -> (1,1),  6 -> (1,2),  7 -> (1,3)  ##
##  8 -> (2,0),  9 -> (2,1), 10 -> (2,2), 11 -> (2,3)  ##
#########################################################
def action_idx_to_position(idx):
    return idx // 4, idx % 4


#########################################################
# 게임판 상태의 저장, 출력 그리고 종료 판정을 수행하는 State 클래스   #
#########################################################
class State:
    def __init__(self, board_rows=3, board_cols=4):
        # 게임판 상태는 board_rows * board_cols 크기의 배열로 표현
        # 게임판에서 플레이어는 정수값으로 구분
        # 1 : 선공 플레이어, -1 : 후공 플레이어, 0 : 초기 공백 상태
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.board_size = board_rows * board_cols

        ### [NOTE] ###
        self.data = np.zeros(shape=[board_rows, board_cols], dtype=int)
        ##############

        self.winner = None

    # 현 상태에서 유효한 행동 ID 리스트 반환
    def get_available_actions(self):
        available_actions = []
        if self.is_finish():
            return available_actions

        available_actions = [i for i in range(12) if self.data.flatten()[i] == 0]

        if len(available_actions) == 12:
            available_actions.remove(5)
            available_actions.remove(6)

        return available_actions

    # 플레이어가 종료 상태에 있는지 판단.
    # 플레이어가 게임을 이기거나, 지거나, 비겼다면 True 반환, 그 외는 False 반환
    def is_finish(self):
        data_flat = self.data.flatten()

        # 14개의 승리 조건 3-원소 벡터 및 각 벡터 내 idx 정보
        seq3_list = [[0,  1,  2],  # horizontal
                     [1,  2,  3],
                     [4,  5,  6],
                     [5,  6,  7],
                     [8,  9, 10],
                     [9, 10, 11],
                     [0,  4,  8],  # vertical
                     [1,  5,  9],
                     [2,  6, 10],
                     [3,  7, 11],
                     [0,  5, 10],  # diagonal
                     [1,  6, 11],
                     [2,  5,  8],
                     [3,  6,  9]]

        # 게임이 종료되었는지 체크
        for seq3 in seq3_list:
            if data_flat[seq3[0]] == 0:
                continue

            if data_flat[seq3[0]] == data_flat[seq3[1]] == data_flat[seq3[2]]:
                self.winner = data_flat[seq3[0]]
                return True

        # 게임이 계속 지속되어야 하는지 체크
        for i in data_flat:
            if i == 0:
                return False

        # 무승부
        self.winner = 0
        return True

    # 게임판 상태 출력
    def get_state_as_board(self):
        board_str = "┌───┬───┬───┬───┐\n"
        for i in range(self.board_rows):
            board_str += '│'
            for j in range(self.board_cols):
                board_str += ' ' + PLAYER_TO_SYMBOL[int(self.data[i, j])] + ' │'
            board_str += '\n'

            if i < self.board_rows - 1:
                board_str += '├───┼───┼───┼───┤\n'
            else:
                board_str += '└───┴───┴───┴───┘'

        return board_str

    def __str__(self):
        return str([''.join(['O' if x == 1 else 'X' if x == -1 else '-' for x in y]) for y in self.data])


################################################################
# 플레이어 1,2 간의 게임 진행을 담당하는 Env 클래스
class TicTacToe343(MultiAgentEnv):
    def __init__(self, env_config: EnvContext = None):
        super().__init__()
        self.__version__ = "0.0.1"
        if env_config is None:
            env_config = {
                "board_rows": 3,
                "board_cols": 4
            }
        self.env_config = env_config

        self.BOARD_SIZE = self.env_config["board_rows"] * self.env_config["board_cols"]
        self.current_state = None  # 현재 상태 관리
        self.current_agent_int = None  # 현재 에이전트(플레이어) 관리
        self.current_agent = None
        self.ALL_ACTIONS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        self.action_space = spaces.Discrete(n=len(self.ALL_ACTIONS))
        self.observation_space = spaces.Box(0.0, 2.0, (self.BOARD_SIZE,), float)

        # 초기 상태 설정
        self.INITIAL_STATE = State(board_rows=self.env_config["board_rows"], board_cols=self.env_config["board_cols"])

        self.steps = None

    def _next_agent(self):
        if self.current_agent_int == PLAYER_1_INT:
            return 'O'
        elif self.current_agent_int == PLAYER_2_INT:
            return 'X'
        else:
            raise ValueError()

    def reset(self, **kwargs):
        self.steps = 0

        self.current_agent_int = PLAYER_1_INT
        self.current_agent = self._next_agent()

        self.current_state = self.INITIAL_STATE

        observations = {
            self.current_agent: self.current_state.data.flatten()
        }

        return observations

    # 게임 진행을 위해 각 플레이어의 착수 때 마다 호출
    def step(self, actions=None):
        assert len(actions.keys()) == 1, "Enter an action for a single player"
        agent = list(actions.keys())[0]
        agent_action = list(actions.values())[0]
        assert agent == self.current_agent, "agent: {0}, self.current_agent: {1}".format(agent, self.current_agent)

        if agent_action in self.current_state.get_available_actions():
            self.steps += 1

            # 플레이어의 행동에 의한 다음 상태 갱신
            position = action_idx_to_position(agent_action)

            next_state = self.get_new_state(
                i=position[0], j=position[1],
                state_data=self.current_state.data,
                player_int=self.current_agent_int
            )

            finish = next_state.is_finish()

            if finish:
                self.current_state = next_state

                next_observations = {
                    'O': next_state.data.flatten(),
                    'X': next_state.data.flatten()
                }

                if next_state.winner == PLAYER_1_INT:
                    rewards = {
                        'O': 1.0, 'X': -1.0
                    }
                elif next_state.winner == PLAYER_2_INT:
                    rewards = {
                        'O': -1.0, 'X': 1.0
                    }
                else:
                    rewards = {
                        'O': 0.0, 'X': 0.0
                    }

                infos = {
                    'O': {'winner': next_state.winner},
                    'X': {'winner': next_state.winner},
                }
            else:
                self.current_state = next_state

                if self.current_agent_int == PLAYER_1_INT:
                    self.current_agent_int = PLAYER_2_INT
                elif self.current_agent_int == PLAYER_2_INT:
                    self.current_agent_int = PLAYER_1_INT
                else:
                    raise ValueError()

                self.current_agent = self._next_agent()

                next_observations = {self.current_agent: self.current_state.data.flatten()}
                rewards = {self.current_agent: 0.0}
                infos = {
                    self.current_agent: {}
                }
        else:
            # 이미 돌이 두어진 자리에 다시 돌을 두려고 시도할 때, 동일한 에이전트에게 정보 전달
            next_observations = {self.current_agent: self.current_state.data.flatten()}
            rewards = {self.current_agent: -10.0}
            finish = False
            infos = {
                self.current_agent: {}
            }

        assert self.steps <= 12

        dones = {"__all__": finish}

        return next_observations, rewards, dones, infos

    def render(self):
        print(self.current_state.get_state_as_board())

    def get_new_state(self, i, j, state_data, player_int):
        new_state = State(board_rows=self.env_config["board_rows"], board_cols=self.env_config["board_cols"])

        # 주어진 상태의 게임판 상황 복사
        new_state.data = np.copy(state_data)

        # 플레이어의 행동(i, j 위치에 표시) 반영
        new_state.data[i, j] = player_int

        return new_state

    def print_board_idx(self):
        print()
        print("[[[Tic-Tac-Toe 보드 내 각 셀을 선택할 때 다음 숫자 키패드를 사용하세요.]]]")
        for i in range(self.env_config["board_rows"]):
            print('-------------')
            out = '| '
            for j in range(self.env_config["board_cols"]):
                out += str(position_to_action_idx(i, j)) + ' | '
            print(out)
        print('-------------')


def main():
    env = TicTacToe343()

    observations = env.reset()

    env.render()

    agents = Dummy_Agents()

    dones = dict()
    dones["__all__"] = False
    total_steps = 0

    while not dones["__all__"]:
        total_steps += 1

        actions = agents.get_actions(observations)

        next_observations, rewards, dones, infos = env.step(actions)

        print("[Step: {0:>2}] observations: {1}, actions: {2}, next_observations: {3}, rewards: {4}, dones: {5}, infos: {6}".format(
            total_steps, observations, actions, next_observations, rewards, dones, infos, total_steps
        ), end="\n\n")

        env.render()

        observations = next_observations
        time.sleep(0.1)


if __name__ == "__main__":
    main()
