import gym
from gym import spaces
import numpy as np

from .elevator_base import Elevator, ElevatorState, Request, TIME_PER_FLOOR

MAX_PEOPLE = 50

REWARD_PER_TIMESTEP = -1
REWARD_PER_SUCCESS = 0


class ElevatorV1Env(gym.Env):
    metadata = {"render.modes": ["human"], "video.frames_per_second": 15}

    def __init__(self, num_elevators: int = 1, num_floors: int = 3, episode_len: int = 200):
        self.num_elevators: int = num_elevators
        self.num_floors: int = num_floors
        self.episode_len: int = episode_len

        obs_space = []
        for _ in range(self.num_elevators):
            obs_space.append(self.num_floors)     # floor num
            obs_space.append(self.num_floors)     # target floor
            obs_space.append(TIME_PER_FLOOR + 1)  # time to next floor
            obs_space.append(3)                   # direction
            for _ in range(self.num_floors):
                obs_space.append(MAX_PEOPLE)      # num people in the elevator requesting each floor

        for _ in range(self.num_floors):
            obs_space.append(MAX_PEOPLE)          # num people waiting on each floor
        self.observation_space: spaces.Space = spaces.MultiDiscrete(obs_space)
        # self.observation_space: spaces.Space = spaces.Dict({
        #     'floor': spaces.MultiDiscrete([self.num_floors for _ in range(self.num_elevators)]),
        #     'target_floor': spaces.MultiDiscrete([self.num_floors for _ in range(self.num_elevators)]),
        #     'time_to_next_floor': spaces.MultiDiscrete([TIME_PER_FLOOR + 1 for _ in range(self.num_elevators)]),
        #     'direction': spaces.MultiDiscrete([3 for _ in range(self.num_elevators)]),
        #     'num_people': spaces.MultiDiscrete([MAX_PEOPLE for _ in range(self.num_floors * self.num_elevators)]),
        #
        #     'waiting': spaces.MultiDiscrete([MAX_PEOPLE for _ in range(self.num_floors)])
        # })
        self.action_space: spaces.Space = spaces.MultiDiscrete(
            [self.num_floors for _ in range(self.num_elevators)]  # target floor for each elevator
        )

        self._init_state()
        self.rng = np.random.default_rng()

    def _init_state(self):
        self.elevators: list[Elevator] = [Elevator(0, 0, self.num_floors) for _ in range(self.num_elevators)]  # all elevators at ground
        self.unassigned_requests: dict[int, set[Request]] = {i: set() for i in range(self.num_floors)}

        self.t = 0

        self.num_dropped_off = 0
        self.num_total_requests = 0

    def reset(self):
        self._init_state()

        return self.get_obs()

    def get_obs(self):
        obs = []
        for i in range(self.num_elevators):
            obs.append(self.elevators[i].floor)  # floor num
            obs.append(self.elevators[i].target_floor)  # target floor
            obs.append(self.elevators[i].time_to_next_floor)  # time to next floor
            obs.append(self.elevators[i].state)  # direction
            for j in range(self.num_floors):
                obs.append(len(self.elevators[i].requests.get(j, [])))  # num people in the elevator requesting each floor

        for i in range(self.num_floors):
            obs.append(len(self.unassigned_requests.get(i, [])))  # num people waiting on each floor

        # print(f"{obs=}")
        # print(f"{self.observation_space=}")
        assert self.observation_space.contains(np.array(obs))
        return np.array(obs)
        # return {
        #     'floor': np.array([self.elevators[i].floor for i in range(self.num_elevators)]),
        #     'target_floor': np.array([self.elevators[i].target_floor for i in range(self.num_elevators)]),
        #     'time_to_next_floor': np.array([self.elevators[i].time_to_next_floor for i in range(self.num_elevators)]),
        #     'direction': np.array([self.elevators[i].state - 1 for i in range(self.num_elevators)]),
        #     'num_people': np.array([len(self.elevators[i].requests.get(j, [])) for j in range(self.num_floors) for i in range(self.num_elevators)]),
        #     'waiting': np.array([len(self.unassigned_requests.get(j, [])) for j in range(self.num_floors)])
        # }

    def step(self, action: np.ndarray):
        """Returns (state, reward, done, info)"""

        # handle action
        assert self.action_space.contains(action), f"Invalid action {action} for space {self.action_space}"
        for i in range(action.shape[0]):
            self.elevators[i].target_floor = action[i]

        # update elevators, calculate reward
        reward = 0
        for elevator in self.elevators:
            elevator.update_state()

            if elevator.state == ElevatorState.IDLE:
                # release passengers, positive reward per completed request
                num_released_requests = elevator.batch_remove_requests(elevator.floor)
                reward += REWARD_PER_SUCCESS * num_released_requests
                self.num_dropped_off += num_released_requests

                # add waiting passengers
                for request in self.unassigned_requests[elevator.floor]:
                    elevator.add_request(request)
                self.unassigned_requests[elevator.floor].clear()

            # negative reward per passenger riding
            reward += REWARD_PER_TIMESTEP * elevator.num_passengers()

        for requests_set in self.unassigned_requests.values():
            # negative reward per unassigned request
            reward += REWARD_PER_TIMESTEP * len(requests_set)

        # add new requests
        if self.rng.random() < 0.3:
            starting_floor = self.rng.integers(self.num_floors)
            target_floor = self.rng.integers(self.num_floors - 1)
            if target_floor >= starting_floor:
                target_floor += 1

            self.unassigned_requests[starting_floor].add(Request(self.t, target_floor))
            self.num_total_requests += 1

        self.t += 1

        # print(f"{self.t = }, {reward = }")
        # print(self.elevators[0])
        # print(self.unassigned_requests)

        done = self.t > self.episode_len

        return self.get_obs(), reward, done, {}
