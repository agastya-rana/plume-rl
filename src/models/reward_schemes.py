import enum
from typing import Protocol, Any

from src.models.goals import GoalZone
from src.models.odor_histories import OdorHistory


@enum.unique
class RewardSchemeEnum(enum.IntEnum):
    SIMPLE_ODOR_HISTORY = enum.auto()
    GOAL_ZONE = enum.auto()


class RewardScheme(Protocol):

    def get_reward(self, **kwargs: Any) -> float:
        pass


class SimpleOdorHistoryRewardScheme:

    def __init__(self, odor_history: OdorHistory):
        self.odor_history = odor_history

    def get_reward(self) -> float:
        return self.odor_history.value[-1]


class GoalZoneRewardScheme:

    def __init__(self, goal_zone: GoalZone, test_position):
        self.goal_zone = goal_zone
        self.test_position = test_position

    def get_reward(self) -> float:
        if self.goal_zone.is_in_goal_zone(test_position=self.test_position):
            return 1
        else:
            return 0
