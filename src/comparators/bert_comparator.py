from bert_score import score
from typing import List
from .base_comparator import BaseComparator
from ..models import ComparisonResult
from config import BERTConfig

class BERTComparator(BaseComparator):
    def __init__(self, config: BERTConfig):
        """Initialize the BERT comparator with configuration"""
        self.config = config

    def compare_scenarios(self, 
                        req_id: str,
                        ai_scenario: str, 
                        manual_scenario: str) -> ComparisonResult:
        """Compare AI-generated and manual scenarios using BERTScore"""
        # Extract steps
        ai_steps = self._extract_scenario_steps(ai_scenario)
        manual_steps = self._extract_scenario_steps(manual_scenario)
        
        # Convert steps to text
        ai_text = ' '.join(ai_steps)
        manual_text = ' '.join(manual_steps)
        
        # Calculate BERTScore
        P, R, F1 = score(
            [ai_text], 
            [manual_text], 
            lang=self.config.language,
            batch_size=self.config.batch_size,
            nthreads=self.config.nthreads,
            verbose=False
        )
        
        # Create comparison result
        return ComparisonResult(
            ai_scenario_num=req_id,
            manual_scenario_num=req_id,
            ai_scenario=ai_scenario,
            manual_scenario=manual_scenario,
            overall_similarity=F1.item() * 100  # Convert to percentage
        )

