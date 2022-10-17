import gym
from gym import spaces
import numpy as np
from dataclasses import dataclass

MAX_PEOPLE = 50


@dataclass(frozen=True)
class Request:
    time_requested: int
    target_floor: int


class Elevator:
    def __init__(self, floor: int, target_floor: int):
        self.floor: int = floor
        self.target_floor: int = target_floor
        self.time_to_next_floor = 0  # 0 => at floor, > 0 => moving
        self.requests: dict[int, set[Request]] = {}

    def add_request(self, request: Request):
        if request.target_floor not in self.requests:
            self.requests[request.target_floor] = set()
        self.requests[request.target_floor].add(request)

    def batch_add_requests(self, requests: set[Request], floor: int | None = None):
        if floor is None:
            floor = self.floor
        if floor not in self.requests:
            self.requests[floor] = set()
        self.requests[floor].update(requests)

    def remove_request(self, request: Request):
        if request not in self.requests[request.target_floor]:
            return
        self.requests[request.target_floor].remove(request)

    def batch_remove_requests(self, floor: int | None = None) -> int:
        if floor is None:
            floor = self.floor
        if floor not in self.requests:
            return 0
        num_requests = len(self.requests[self.floor])
        self.requests[self.floor].clear()
        return num_requests

    def num_passengers(self):
        n = 0
        for requests_set in self.requests.values():
            n += len(requests_set)
        return n

    def __repr__(self):
        return f"Elevator[{self.floor=}, {self.target_floor=}, {self.time_to_next_floor=}, {self.requests=}]"


class ElevatorV1Env(gym.Env):
    metadata = {"render.modes": ["human"], "video.frames_per_second": 15}

    def __init__(self, num_elevators: int = 1, num_floors: int = 3):
        self.num_elevators: int = num_elevators
        self.num_floors: int = num_floors

        self.observation_space: spaces.Space = spaces.MultiDiscrete(
            [self.num_floors for _ in range(self.num_elevators)] +  # positions of each elevator
            [MAX_PEOPLE + 1 for _ in range(self.num_floors)]        # number of people on each floor
        )
        self.action_space: spaces.Space = spaces.MultiDiscrete(
            [self.num_floors for _ in range(self.num_elevators)]  # target floor for each elevator
        )

        self.elevators: list[Elevator] = [Elevator(0, 0) for _ in range(self.num_elevators)]  # all elevators at ground
        self.unassigned_requests: dict[int, set[Request]] = {i: set() for i in range(self.num_floors)}

        self.t = 0
        self.rng = np.random.default_rng()

    def step(self, action: np.ndarray):
        # handle action
        assert self.action_space.contains(action), f"Invalid action {action} for space {self.action_space}"
        for i in range(action.shape[0]):
            self.elevators[i].target_floor = action[i]

        # update elevators, calculate reward
        reward = 0
        for elevator in self.elevators:
            if elevator.floor != elevator.target_floor:
                elevator.time_to_next_floor = max(0, elevator.time_to_next_floor - 1)
            if elevator.time_to_next_floor == 0:
                # update elevator floor
                if elevator.floor != elevator.target_floor:
                    elevator.floor += 1 if elevator.target_floor > elevator.floor else -1

                # release passengers, positive reward per completed request
                num_released_requests = elevator.batch_remove_requests(elevator.floor)
                reward += 100 * num_released_requests

                # add waiting passengers
                for request in self.unassigned_requests[elevator.floor]:
                    elevator.add_request(request)
                self.unassigned_requests[elevator.floor].clear()
                if elevator.num_passengers() > 0:
                    elevator.time_to_next_floor = 5

            # negative reward per passenger riding
            reward += -1 * elevator.num_passengers()

        for requests_set in self.unassigned_requests.values():
            # negative reward per unassigned request
            reward += -1 * len(requests_set)

        # add new requests
        if self.rng.random() < 0.3:
            starting_floor = self.rng.integers(self.num_floors)
            target_floor = self.rng.integers(self.num_floors)
            while target_floor == starting_floor:
                target_floor = self.rng.integers(self.num_floors)

            self.unassigned_requests[starting_floor].add(Request(self.t, target_floor))

        self.t += 1

        return reward

