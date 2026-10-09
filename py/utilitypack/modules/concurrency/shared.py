import enum


class TaskState(enum.Enum):
    WAITING = enum.auto()
    SUCCEEDED = enum.auto()
    FAILED = enum.auto()
