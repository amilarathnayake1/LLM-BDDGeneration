from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import List, Tuple
from .base_comparator import BaseComparator
from ..models import ComparisonResult
from config import BLEUConfig

class BLEUComparator(BaseComparator):
    def __init__(self, config: BLEUConfig):
        """Initialize the BLEU comparator with configuration"""
        self.config = config
        self.smoothing = SmoothingFunction().method1

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        return text.lower().split()

    def _clean_scenario_text(self, text: str) -> str:
        """Clean scenario text by removing table structures and extra whitespace"""
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            # Skip table headers, dividers, and empty lines
            if (not line or 
                '|' in line or 
                line.startswith('Feature:') or 
                line.startswith('Scenario:')):
                continue
            # Remove leading pipe and trim
            if line.startswith('|'):
                line = line.split('|')[1].strip()
            lines.append(line)
        return ' '.join(lines)

    def _extract_main_steps(self, scenario: str) -> List[str]:
        """Extract main Given/When/Then steps without table content"""
        main_steps = []
        current_step = []
        is_in_table = False
        
        for line in scenario.split('\n'):
            line = line.strip()
            
            # Skip empty lines and Feature/Scenario lines
            if not line or line.startswith('Feature:') or line.startswith('Scenario:'):
                continue
                
            # Check if we're entering or leaving a table
            if '|' in line:
                is_in_table = True
                continue
                
            # If we're not in a table and line starts with a step keyword
            if not is_in_table and any(line.startswith(keyword) for keyword in ['Given', 'When', 'Then', 'And']):
                if current_step:
                    main_steps.append(' '.join(current_step))
                current_step = [line]
            elif not is_in_table and current_step:
                current_step.append(line)
                
            # Reset table flag if we've moved past the table
            if line and not '|' in line:
                is_in_table = False
        
        # Add the last step if exists
        if current_step:
            main_steps.append(' '.join(current_step))
            
        return main_steps

    def compare_scenarios(self, 
                        req_id: str,
                        ai_scenario: str, 
                        manual_scenario: str) -> ComparisonResult:
        """Compare AI-generated and manual scenarios using BLEU score"""
        # Extract main steps without tables
        ai_steps = self._extract_main_steps(ai_scenario)
        manual_steps = self._extract_main_steps(manual_scenario)
        
        # Calculate BLEU score for main steps
        step_scores = []
        for ai_step, manual_step in zip(ai_steps, manual_steps):
            # Tokenize cleaned steps
            ai_tokens = self._tokenize(self._clean_scenario_text(ai_step))
            manual_tokens = self._tokenize(self._clean_scenario_text(manual_step))
            
            # Calculate BLEU score
            score = sentence_bleu(
                [manual_tokens],
                ai_tokens,
                smoothing_function=self.smoothing,
                weights=self.config.weights
            )
            step_scores.append(score)
        
        # Calculate overall score
        overall_score = sum(step_scores) / len(step_scores) if step_scores else 0
        
        # Create detailed comparison result
        return ComparisonResult(
            ai_scenario_num=req_id,
            manual_scenario_num=req_id,
            ai_scenario=ai_scenario,
            manual_scenario=manual_scenario,
            overall_similarity=overall_score * 100  # Convert to percentage
        )

    def _print_comparison_details(self, ai_steps: List[str], manual_steps: List[str]) -> None:
        """Print detailed comparison information for debugging"""
        print("\nStep Comparison Details:")
        print("-" * 80)
        for i, (ai, manual) in enumerate(zip(ai_steps, manual_steps)):
            print(f"\nStep {i+1}:")
            print(f"AI     : {ai}")
            print(f"Manual : {manual}")
            
            # Calculate and print individual step score
            ai_tokens = self._tokenize(self._clean_scenario_text(ai))
            manual_tokens = self._tokenize(self._clean_scenario_text(manual))
            score = sentence_bleu(
                [manual_tokens],
                ai_tokens,
                smoothing_function=self.smoothing,
                weights=self.config.weights
            )
            print(f"Score  : {score * 100:.2f}%")