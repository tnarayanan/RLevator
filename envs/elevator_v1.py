import gym
from gym import spaces
import numpy as np

from .elevator_base import Elevator, ElevatorState, Request, TIME_PER_FLOOR

MAX_PEOPLE = 50

REWARD_PER_TIMESTEP = -1
REWARD_PER_SUCCESS = 100


class ElevatorV1Env(gym.Env):
    metadata = {"render.modes": ["human"], "video.frames_per_second": 15}

    def __init__(self, num_elevators: int = 1, num_floors: int = 3):
        self.num_elevators: int = num_elevators
        self.num_floors: int = num_floors

        # self.observation_space: spaces.Space = spaces.MultiDiscrete(
        #     [self.num_floors for _ in range(self.num_elevators)] +  # positions of each elevator
        #     [MAX_PEOPLE + 1 for _ in range(self.num_floors)]        # number of people on each floor
        # )
        self.observation_space: spaces.Space = spaces.Dict({
            f'elevator{i}': spaces.Dict({
                'floor': spaces.Discrete(self.num_floors),
                'target_floor': spaces.Discrete(self.num_floors),
                'time_to_next_floor': spaces.Discrete(TIME_PER_FLOOR + 1),
                'direction': spaces.Discrete(3),
                'num_people': spaces.MultiDiscrete([MAX_PEOPLE for _ in range(self.num_floors)])
            })
            for i in range(self.num_elevators)
        } | {
            'waiting': spaces.MultiDiscrete([MAX_PEOPLE for _ in range(self.num_floors)])
        })
        self.action_space: spaces.Space = spaces.MultiDiscrete(
            [self.num_floors for _ in range(self.num_elevators)]  # target floor for each elevator
        )

        self.elevators: list[Elevator] = [Elevator(0, 0) for _ in range(self.num_elevators)]  # all elevators at ground
        self.unassigned_requests: dict[int, set[Request]] = {i: set() for i in range(self.num_floors)}

        self.t = 0
        self.rng = np.random.default_rng()

    def get_obs(self):
        return {
            f'elevator{i}': {
                'floor': self.elevators[i].floor,
                'target_floor': self.elevators[i].target_floor,
                'time_to_next_floor': self.elevators[i].time_to_next_floor,
                'direction': self.elevators[i].state - 1,
                'num_people': np.array([len(self.elevators[i].requests.get(j, [])) for j in range(self.num_floors)])
            } for i in range(self.num_elevators)
        } | {
            'waiting': np.array([len(self.unassigned_requests.get(j, [])) for j in range(self.num_floors)])
        }

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
            target_floor = self.rng.integers(self.num_floors)
            while target_floor == starting_floor:
                target_floor = self.rng.integers(self.num_floors)

            self.unassigned_requests[starting_floor].add(Request(self.t, target_floor))

        self.t += 1

        print(f"{self.t = }, {reward = }")
        print(self.elevators[0])
        print(self.unassigned_requests)

        done = self.t > 30

        return self.get_obs(), reward, done, {}
