from typing import Dict, List
from difflib import SequenceMatcher
import numpy as np
from .models import BDDScenario
from bert_score import BERTScorer
import config

class BERTScenarioComparator:
    def __init__(self):
        """Initialize the ScenarioComparator with BERTScore."""
        # Initialize BERTScorer with DeBERTa model (state-of-the-art for semantic similarity)
        self.bert_scorer = BERTScorer(
            model_type="microsoft/deberta-xlarge-mnli",
            num_layers=18,
            batch_size=8,
            nthreads=4,
            rescale_with_baseline=True
        )
        self.weights = config.SIMILARITY_WEIGHTS

    def calculate_semantic_similarity(self, steps1: List[str], steps2: List[str]) -> float:
        """Calculate semantic similarity between steps using BERTScore."""
        if not steps1 or not steps2:
            return 0.0
        
        try:
            # Combine steps into single strings
            text1 = ' '.join(steps1)
            text2 = ' '.join(steps2)
            
            # Calculate BERTScore (returns Precision, Recall, and F1)
            P, R, F1 = self.bert_scorer.score([text1], [text2])
            
            # Return F1 score (harmonic mean of precision and recall)
            return float(F1.mean())
            
        except Exception as e:
            print(f"Error calculating BERTScore similarity: {str(e)}")
            return 0.0

    def compare_scenarios(self, scenario1: BDDScenario, scenario2: BDDScenario) -> Dict[str, float]:
        """Compare two BDD scenarios using BERTScore for semantic similarity."""
        # Calculate semantic similarities for each section
        given_similarity = self.calculate_semantic_similarity(scenario1.given, scenario2.given)
        when_similarity = self.calculate_semantic_similarity(scenario1.when, scenario2.when)
        then_similarity = self.calculate_semantic_similarity(scenario1.then, scenario2.then)
        
        # Calculate title similarity
        title_similarity = self.calculate_semantic_similarity([scenario1.title], [scenario2.title])
        
        # Calculate overall similarity (weighted average)
        overall_similarity = (
            self.weights['title'] * title_similarity +
            self.weights['given'] * given_similarity +
            self.weights['when'] * when_similarity +
            self.weights['then'] * then_similarity
        )
        
        return {
            'title_similarity': round(title_similarity, 3),
            'given_similarity': round(given_similarity, 3),
            'when_similarity': round(when_similarity, 3),
            'then_similarity': round(then_similarity, 3),
            'overall_similarity': round(overall_similarity, 3)
        }

    def get_similarity_report(self, ai_scenarios: List[BDDScenario], 
                            manual_scenarios: List[BDDScenario]) -> str:
        """Generate a detailed similarity report comparing all scenarios."""
        report = []
        report.append("BDD Scenarios Semantic Similarity Report (BERTScore)")
        report.append("=" * 50)
        
        all_comparisons = []
        
        for i, ai_scenario in enumerate(ai_scenarios):
            for j, manual_scenario in enumerate(manual_scenarios):
                scores = self.compare_scenarios(ai_scenario, manual_scenario)
                
                report.append(f"\nComparison: AI Scenario {i+1} vs Manual Scenario {j+1}")
                report.append("-" * 50)
                report.append(f"Title Similarity: {scores['title_similarity']:.2%}")
                report.append(f"Given Steps Similarity: {scores['given_similarity']:.2%}")
                report.append(f"When Steps Similarity: {scores['when_similarity']:.2%}")
                report.append(f"Then Steps Similarity: {scores['then_similarity']:.2%}")
                report.append(f"Overall Semantic Similarity: {scores['overall_similarity']:.2%}")
                
                if scores['overall_similarity'] >= config.MIN_SIMILARITY_THRESHOLD:
                    report.append("✓ High semantic similarity detected")
                else:
                    report.append("⚠ Low semantic similarity detected")
                
                # Store comparison results for summary
                all_comparisons.append(scores['overall_similarity'])
        
        # Add summary statistics
        report.append("\nSummary Statistics")
        report.append("-" * 20)
        report.append(f"Average Similarity: {np.mean(all_comparisons):.2%}")
        report.append(f"Maximum Similarity: {np.max(all_comparisons):.2%}")
        report.append(f"Minimum Similarity: {np.min(all_comparisons):.2%}")
        
        return "\n".join(report)

    def save_detailed_results(self, ai_scenarios: List[BDDScenario], 
                            manual_scenarios: List[BDDScenario],
                            output_file: str):
        """Save detailed comparison results to a CSV file."""
        results = []
        
        for i, ai_scenario in enumerate(ai_scenarios):
            for j, manual_scenario in enumerate(manual_scenarios):
                scores = self.compare_scenarios(ai_scenario, manual_scenario)
                
                result = {
                    'ai_scenario_num': i + 1,
                    'manual_scenario_num': j + 1,
                    'ai_scenario': str(ai_scenario),
                    'manual_scenario': str(manual_scenario),
                    **scores
                }
                results.append(result)
        
        # Convert to DataFrame and save
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"Detailed results saved to {output_file}")