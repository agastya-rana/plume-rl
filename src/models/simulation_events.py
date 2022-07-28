import enum
from typing import Callable, Any


class SimulationEvent(enum.Enum):
    RESET = enum.auto()
    STEP = enum.auto()


"""
def subscribe(event_type: SimulationEvent, func: Callable):
    if event_type not in subscribers:
        subscribers[event_type] = []
        subscribers[event_type].append(func)
"""


def process_event(subscribers: dict[SimulationEvent, list[Callable]], event_type: SimulationEvent):
    """
    This is an event handling system that maps events like gym environment RESET onto triggered methods
    For example, RESET can trigger a method like odor_history.clear
    """
    if event_type not in subscribers:
        return
    for func in subscribers[event_type]:
        func()
