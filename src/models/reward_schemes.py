import enum
from typing import Protocol, Any

from src.models.goals import GoalZone
from src.models.odor_senses import OdorHistory, OdorFeatures

ODOR_REWARD_THRESHOLD = 100  # from Nirag


@enum.unique
class RewardSchemeEnum(enum.IntEnum):
    SIMPLE_ODOR_HISTORY = enum.auto()
    SIMPLE_ODOR_FEATURES = enum.auto()
    NIRAG_REWARD = enum.auto()
    GOAL_ZONE = enum.auto()
    Y_MAX = enum.auto()


class RewardScheme(Protocol):

    def get_reward(self, **kwargs: Any) -> float:
        pass


class SimpleOdorHistoryRewardScheme:

    def __init__(self, odor_history: OdorHistory):
        self.odor_history = odor_history

    def get_reward(self) -> float:
        return self.odor_history.value[-1]


class SimpleOdorFeatureRewardScheme:

    def __init__(self, odor_features: OdorFeatures):
        self.odor_features = odor_features

    def get_reward(self) -> float:
        return self.odor_features.concentration


class NiragRewardScheme:

    def __init__(self, odor_features: OdorFeatures):
        self.odor_features = odor_features
        self.threshold = ODOR_REWARD_THRESHOLD  # From Nirag

    def get_reward(self) -> float:
        return 1 * (self.odor_features.concentration > self.threshold)


class YMaxRewardScheme:
    """
    This class gives reward for increasing the y coordinate. Useful for testing/debugging because the expected
    behavior after training is clear
    """

    def __init__(self, y_displacement: float):
        self.y_displacement = y_displacement

    def get_reward(self) -> float:
        return self.y_displacement


class GoalZoneRewardScheme:

    def __init__(self, goal_zone: GoalZone, test_position):
        self.goal_zone = goal_zone
        self.test_position = test_position

    def get_reward(self) -> float:
        if self.goal_zone.is_in_goal_zone(test_position=self.test_position):
            return 1
        else:
            return 0
