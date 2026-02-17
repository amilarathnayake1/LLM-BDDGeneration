from dataclasses import dataclass
from typing import List

@dataclass
class RequirementData:
    id: str
    user_story: str
    requirements: str
    manual_scenario: str

@dataclass
class BDDStep:
    keyword: str  # Given, When, Then, And
    description: str

@dataclass
class BDDScenario:
    feature_name: str
    scenario_name: str
    steps: List[BDDStep]

@dataclass
class ComparisonResult:
    ai_scenario_num: str
    manual_scenario_num: str
    ai_scenario: str
    manual_scenario: str
    overall_similarity: float

    def to_dict(self):
        return {
            'ai_scenario_num': self.ai_scenario_num,
            'manual_scenario_num': self.manual_scenario_num,
            'ai_scenario': self.ai_scenario,
            'manual_scenario': self.manual_scenario,
            'overall_similarity_%': round(self.overall_similarity, 2)
        }