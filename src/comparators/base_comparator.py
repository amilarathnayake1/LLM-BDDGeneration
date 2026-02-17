from abc import ABC, abstractmethod
from typing import List
from ..models import ComparisonResult

class BaseComparator(ABC):
    def _extract_scenario_steps(self, scenario: str) -> List[str]:
        """Extract individual steps from a BDD scenario"""
        steps = []
        for line in scenario.split('\n'):
            line = line.strip()
            if any(line.startswith(keyword) for keyword in ['Given', 'When', 'Then', 'And']):
                steps.append(line)
        return steps

    @abstractmethod
    def compare_scenarios(self, req_id: str, ai_scenario: str, manual_scenario: str) -> ComparisonResult:
        """Compare AI-generated and manual scenarios"""
        pass