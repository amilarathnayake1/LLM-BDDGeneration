from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from .base_comparator import BaseComparator
from ..models import ComparisonResult
from config import SBCSConfig

class SBCSComparator(BaseComparator):
    def __init__(self, config: SBCSConfig):
        """Initialize the SentenceBERT with Cosine Similarity comparator"""
        self.config = config
        self.model = SentenceTransformer(config.model_name, device=config.device)

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
            
            if not line or line.startswith('Feature:') or line.startswith('Scenario:'):
                continue
                
            if '|' in line:
                is_in_table = True
                continue
                
            if not is_in_table and any(line.startswith(keyword) for keyword in ['Given', 'When', 'Then', 'And']):
                if current_step:
                    main_steps.append(' '.join(current_step))
                current_step = [line]
            elif not is_in_table and current_step:
                current_step.append(line)
                
            if line and not '|' in line:
                is_in_table = False
        
        if current_step:
            main_steps.append(' '.join(current_step))
            
        return main_steps

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts using SentenceBERT embeddings"""
        # Generate embeddings
        embedding1 = self.model.encode([text1], convert_to_tensor=True)
        embedding2 = self.model.encode([text2], convert_to_tensor=True)
        
        # Convert to numpy for cosine similarity calculation
        embedding1_np = embedding1.cpu().numpy()
        embedding2_np = embedding2.cpu().numpy()
        
        # Calculate cosine similarity
        similarity = cosine_similarity(embedding1_np, embedding2_np)[0][0]
        
        return similarity

    def compare_scenarios(self, 
                        req_id: str,
                        ai_scenario: str, 
                        manual_scenario: str) -> ComparisonResult:
        """Compare AI-generated and manual scenarios using SentenceBERT and Cosine Similarity"""
        # Extract main steps without tables
        ai_steps = self._extract_main_steps(ai_scenario)
        manual_steps = self._extract_main_steps(manual_scenario)
        
        # Calculate similarity for each step pair
        step_scores = []
        for ai_step, manual_step in zip(ai_steps, manual_steps):
            # Clean and prepare text
            ai_text = self._clean_scenario_text(ai_step)
            manual_text = self._clean_scenario_text(manual_step)
            
            # Calculate similarity
            similarity = self._calculate_similarity(ai_text, manual_text)
            step_scores.append(similarity)
        
        # Calculate overall score
        overall_score = sum(step_scores) / len(step_scores) if step_scores else 0
        
        # Create comparison result
        return ComparisonResult(
            ai_scenario_num=req_id,
            manual_scenario_num=req_id,
            ai_scenario=ai_scenario,
            manual_scenario=manual_scenario,
            overall_similarity=overall_score * 100  # Convert to percentage
        )

    def _print_comparison_details(self, ai_steps: List[str], manual_steps: List[str]) -> None:
        """Print detailed comparison information for debugging"""
        print("\nStep Comparison Details (SentenceBERT + Cosine Similarity):")
        print("-" * 80)
        for i, (ai, manual) in enumerate(zip(ai_steps, manual_steps)):
            print(f"\nStep {i+1}:")
            print(f"AI     : {ai}")
            print(f"Manual : {manual}")
            
            # Calculate and print individual step score
            ai_text = self._clean_scenario_text(ai)
            manual_text = self._clean_scenario_text(manual)
            similarity = self._calculate_similarity(ai_text, manual_text)
            print(f"Score  : {similarity * 100:.2f}%")