import gym
from gym import spaces
import numpy as np

from .elevator_base import Elevator, ElevatorState, Request, TIME_PER_FLOOR

MAX_PEOPLE = 50

REWARD_PER_TIMESTEP = -1
REWARD_PER_SUCCESS = 0
# REWARD_PER_ENERGY = -1


class ElevatorV2Env(gym.Env):
    metadata = {"render.modes": ["human"], "video.frames_per_second": 15}

    def __init__(self, num_elevators_start: int = 1, num_floors_start: int = 3, curriculum: bool = False, num_elevators_end: int = -1, num_floors_end: int = -1, episode_len: int = 200, random_seed: int = 0, request_prob: float = 0.3):
        self.num_elevators: int = num_elevators_start
        self.num_floors: int = num_floors_start
        self.curriculum = curriculum

        self.num_floors_start = num_floors_start
        self.num_elevators_start = num_elevators_start
        self.num_elevators_end = num_elevators_end if self.curriculum else self.num_elevators_start
        self.num_floors_end = num_floors_end if self.curriculum else self.num_floors_start

        self.episode_len: int = episode_len

        obs_space = []
        obs_space.append(self.num_elevators_end + 1)  # cur num elevators
        obs_space.append(self.num_floors_end + 1)  # cur num floors
        for _ in range(self.num_elevators_end):
            obs_space.append(self.num_floors_end)     # floor num
            obs_space.append(self.num_floors_end)     # target floor
            obs_space.append(TIME_PER_FLOOR + 1)  # time to next floor
            obs_space.append(3)                   # direction
            for _ in range(self.num_floors_end):
                obs_space.append(MAX_PEOPLE)      # num people in the elevator requesting each floor

        for _ in range(self.num_floors_end):
            obs_space.append(MAX_PEOPLE)          # num people waiting on each floor
        self.observation_space: spaces.Space = spaces.MultiDiscrete(obs_space)

        self.action_space: spaces.Space = spaces.MultiDiscrete(
            [self.num_floors_end for _ in range(self.num_elevators_end)]  # target floor for each elevator
        )

        self.request_prob = request_prob

        self._init_state()
        self.rng = np.random.default_rng(random_seed)

        self.num_dropped_off = 0
        self.num_total_requests = 0

        self._reset_history()

    def _init_state(self):
        self.elevators: list[Elevator] = [Elevator(0, 0, self.num_floors) for _ in range(self.num_elevators)]  # all elevators at ground
        self.unassigned_requests: dict[int, list[Request]] = {i: list() for i in range(self.num_floors)}

        self.t = 0
        self.total_t = 0

        self.num_dropped_off = 0
        self.num_total_requests = 0

    def _reset_history(self):
        self.history_len = 10
        self.dropped_off_history = []
        self.requests_history = []

    def _update_curriculum(self):
        if self.num_elevators < self.num_elevators_end and (self.num_floors - self.num_floors_start) / (self.num_floors_end - self.num_floors_start) >= (self.num_elevators + 1 - self.num_elevators_start) / (self.num_elevators_end - self.num_elevators_start):
            print(f"Updating curriculum: num_elevators {self.num_elevators} -> {self.num_elevators + 1}")
            self.num_elevators += 1
            self._reset_history()
        elif self.num_floors < self.num_floors_end:
            print(f"Updating curriculum: num_floors {self.num_floors} -> {self.num_floors + 1}")
            self.num_floors += 1
            self._reset_history()

    def reset(self, override_curriculum=False, log=False):
        self.dropped_off_history.append(self.num_dropped_off)
        self.requests_history.append(self.num_total_requests)
        if len(self.dropped_off_history) > self.history_len:
            self.dropped_off_history.pop(0)
            self.requests_history.pop(0)

        # print(f"{self.num_floors}: {self.num_dropped_off}/{self.num_total_requests}, history = {sum(self.dropped_off_history) / sum(self.requests_history)}")

        # check if we should update curriculum
        if self.num_floors <= 4:
            threshold = 0.7
        elif self.num_floors <= 6:
            threshold = 0.5
        elif self.num_floors <= 8:
            threshold = 0.375
        else:
            threshold = 0.3
        if not override_curriculum and len(self.dropped_off_history) >= self.history_len and sum(self.dropped_off_history) > threshold * sum(self.requests_history):
            self._update_curriculum()

        self._init_state()

        return self.get_obs()

    def get_obs(self):
        obs = []
        obs.append(self.num_elevators)
        obs.append(self.num_floors)
        for i in range(self.num_elevators):
            obs.append(self.elevators[i].floor)  # floor num
            obs.append(self.elevators[i].target_floor)  # target floor
            obs.append(self.elevators[i].time_to_next_floor)  # time to next floor
            obs.append(self.elevators[i].state)  # direction
            for j in range(self.num_floors):
                obs.append(len(self.elevators[i].requests.get(j, [])))  # num people in the elevator requesting each floor

            # fill in rest of floors
            for _ in range(self.num_floors_end - self.num_floors):
                obs.append(0)

        # fill in rest of elevators
        for _ in range(self.num_elevators_end - self.num_elevators):
            obs.append(0)  # floor num
            obs.append(0)  # target floor
            obs.append(0)  # time to next floor
            obs.append(1)  # direction
            for _ in range(self.num_floors_end):
                obs.append(0)

        for i in range(self.num_floors):
            obs.append(len(self.unassigned_requests.get(i, [])))  # num people waiting on each floor

        # fill in rest of floors
        for _ in range(self.num_floors_end - self.num_floors):
            obs.append(0)

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
        for i in range(self.num_elevators):
            self.elevators[i].target_floor = min(self.num_floors - 1, action[i])

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

        for requests_list in self.unassigned_requests.values():
            # negative reward per unassigned request
            reward += REWARD_PER_TIMESTEP * len(requests_list)

        # add new requests
        if self.rng.random() < self.request_prob:
            starting_floor = self.rng.integers(self.num_floors)
            target_floor = self.rng.integers(self.num_floors - 1)
            if target_floor >= starting_floor:
                target_floor += 1

            self.unassigned_requests[starting_floor].append(Request(self.t, target_floor))
            self.num_total_requests += 1

        self.t += 1
        self.total_t += 1

        # print(f"{self.t = }, {reward = }")
        # print(self.elevators[0])
        # print(self.unassigned_requests)

        done = self.t > self.episode_len

        return self.get_obs(), reward, done, {}
