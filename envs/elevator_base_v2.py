from dataclasses import dataclass
from enum import IntEnum


@dataclass(frozen=True)
class Request:
    time_requested: int
    target_floor: int


class ElevatorState(IntEnum):
    MOVING_DOWN = 0
    IDLE = 1
    MOVING_UP = 2


class Elevator:
    def __init__(self, floor: int, target_floor: int, num_floors: int):
        self.floor: int = floor
        self.target_floor: int = target_floor
        self.state: ElevatorState = ElevatorState.IDLE
        self.requests: dict[int, list[Request]] = {}

        self.num_floors = num_floors

    def add_request(self, request: Request):
        if request.target_floor not in self.requests:
            self.requests[request.target_floor] = []
        self.requests[request.target_floor].append(request)

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
        for requests_list in self.requests.values():
            n += len(requests_list)
        return n

    def update_state(self):
        if self.target_floor == self.floor:
            self.state = ElevatorState.IDLE
        elif self.target_floor < self.floor:
            self.state = ElevatorState.MOVING_DOWN
        else:
            self.state = ElevatorState.MOVING_UP

        if self.state == ElevatorState.MOVING_UP and self.floor < self.num_floors - 1:
            self.floor += 1
        elif self.state == ElevatorState.MOVING_DOWN and self.floor > 0:
            self.floor -= 1

        if self.target_floor == self.floor:
            self.state = ElevatorState.IDLE
        elif self.target_floor < self.floor:
            self.state = ElevatorState.MOVING_DOWN
        else:
            self.state = ElevatorState.MOVING_UP

        # print(self)

    def __repr__(self):
        return f"Elevator[{self.floor=}, {self.target_floor=}, {self.state=}, {self.requests=}]"
