"""
Research Framework Integration for DeepSeek BDD Evaluator

This module integrates the DeepSeek LLM evaluation with the existing
research framework, providing statistical analysis and correlation
studies as described in the research methodology.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, pearsonr
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BootstrapResult:
    """Bootstrap analysis result"""
    mean: float
    std: float
    confidence_interval: Tuple[float, float]
    percentiles: Dict[str, float]

class BootstrapAnalysis:
    """
    Bootstrap confidence interval analysis as used in the research framework
    """
    
    def __init__(self, n_bootstrap: int = 1000, confidence_level: float = 0.95):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        
    def bootstrap_metric(self, data: np.ndarray, metric_func: callable = np.mean) -> BootstrapResult:
        """
        Perform bootstrap analysis on a metric
        
        Args:
            data: Input data array
            metric_func: Function to compute metric (default: mean)
            
        Returns:
            BootstrapResult with statistics and confidence intervals
        """
        n = len(data)
        bootstrap_values = []
        
        # Generate bootstrap samples
        for _ in range(self.n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            bootstrap_values.append(metric_func(bootstrap_sample))
        
        bootstrap_values = np.array(bootstrap_values)
        
        # Calculate confidence interval
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_values, lower_percentile)
        ci_upper = np.percentile(bootstrap_values, upper_percentile)
        
        # Calculate percentiles
        percentiles = {
            'p25': np.percentile(bootstrap_values, 25),
            'p50': np.percentile(bootstrap_values, 50),
            'p75': np.percentile(bootstrap_values, 75),
            'p90': np.percentile(bootstrap_values, 90),
            'p95': np.percentile(bootstrap_values, 95)
        }
        
        return BootstrapResult(
            mean=np.mean(bootstrap_values),
            std=np.std(bootstrap_values),
            confidence_interval=(ci_lower, ci_upper),
            percentiles=percentiles
        )

class ResearchIntegration:
    """
    Integration class for incorporating DeepSeek evaluation results
    into the existing research framework
    """
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path) if config_path else {}
        self.bootstrap = BootstrapAnalysis(
            n_bootstrap=self.config.get('research_integration', {}).get(
                'statistical_analysis', {}).get('bootstrap_iterations', 1000),
            confidence_level=self.config.get('research_integration', {}).get(
                'statistical_analysis', {}).get('confidence_level', 0.95)
        )
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {}
    
    def map_deepseek_to_human_categories(self, deepseek_results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Map DeepSeek ratings to human evaluation categories used in research
        
        Args:
            deepseek_results_df: DataFrame with DeepSeek evaluation results
            
        Returns:
            DataFrame with mapped categories
        """
        # Mapping based on research framework (updated for 5-point scale)
        category_mapping = {
            5: "Instrumental",  # Excellent - Perfect capture
            4: "Highly Helpful", # Good - Strong alignment with minor issues
            3: "Helpful",       # Moderate - Informative but incomplete
            2: "Somewhat Helpful", # Below Average - Limited alignment
            1: "Misleading",    # Poor - Unrelated to requirement
            -1: "Unavailable"   # Error cases
        }
        
        df = deepseek_results_df.copy()
        
        # Handle both column name formats
        rating_column = 'LLM Evaluator Rating' if 'LLM Evaluator Rating' in df.columns else 'deepseek_rating'
        
        df['human_equivalent_category'] = df[rating_column].map(category_mapping)
        
        # Add numerical mapping for statistical analysis
        numerical_mapping = {
            "Instrumental": 6,
            "Highly Helpful": 5,
            "Helpful": 4,
            "Somewhat Helpful": 3,
            "Misleading": 2,
            "Unavailable": 1
        }
        df['human_equivalent_score'] = df['human_equivalent_category'].map(numerical_mapping)
        
        return df
    
    def calculate_correlation_with_automated_metrics(self, 
                                                   combined_df: pd.DataFrame) -> Dict:
        """
        Calculate correlations between DeepSeek ratings and automated metrics
        
        Args:
            combined_df: DataFrame with DeepSeek ratings and automated metrics
            
        Returns:
            Dictionary with correlation results
        """
        correlation_results = {}
        
        # Metrics to correlate with (from research framework)
        automated_metrics = ['BLEU', 'ROUGE-L', 'BERTScore', 'METEOR', 'SBCS', 'SBED', 'USECS']
        
        # DeepSeek ratings (already aligned with automated metrics where higher = better)
        # Note: DeepSeek now uses 5=best, 1=worst, which aligns with higher automated metrics being better
        rating_column = 'LLM Evaluator Rating' if 'LLM Evaluator Rating' in combined_df.columns else 'deepseek_rating'
        
        # Check if the rating column exists
        if rating_column not in combined_df.columns:
            logger.error(f"Rating column '{rating_column}' not found. Available columns: {list(combined_df.columns)}")
            return {}
        
        deepseek_scores = combined_df[rating_column]
        
        for metric in automated_metrics:
            if metric in combined_df.columns:
                try:
                    # Remove rows with missing values
                    valid_mask = ~(combined_df[metric].isna() | combined_df[rating_column].isna())
                    if valid_mask.sum() < 3:  # Need at least 3 points for correlation
                        continue
                        
                    metric_values = combined_df.loc[valid_mask, metric]
                    valid_deepseek_scores = deepseek_scores.loc[valid_mask]
                    
                    # Spearman correlation (as used in research)
                    spearman_corr, spearman_p = spearmanr(valid_deepseek_scores, metric_values)
                    
                    # Pearson correlation
                    pearson_corr, pearson_p = pearsonr(valid_deepseek_scores, metric_values)
                    
                    # Bootstrap confidence intervals for correlation
                    bootstrap_correlations = []
                    for _ in range(1000):
                        indices = np.random.choice(len(metric_values), size=len(metric_values), replace=True)
                        boot_deepseek = valid_deepseek_scores.iloc[indices]
                        boot_metric = metric_values.iloc[indices]
                        boot_corr, _ = spearmanr(boot_deepseek, boot_metric)
                        if not np.isnan(boot_corr):
                            bootstrap_correlations.append(boot_corr)
                    
                    if bootstrap_correlations:
                        ci_lower = np.percentile(bootstrap_correlations, 2.5)
                        ci_upper = np.percentile(bootstrap_correlations, 97.5)
                    else:
                        ci_lower = ci_upper = np.nan
                    
                    correlation_results[metric] = {
                        'spearman_correlation': spearman_corr,
                        'spearman_p_value': spearman_p,
                        'spearman_significant': spearman_p < 0.05,
                        'pearson_correlation': pearson_corr,
                        'pearson_p_value': pearson_p,
                        'pearson_significant': pearson_p < 0.05,
                        'sample_size': len(valid_deepseek_scores),
                        'bootstrap_ci_lower': ci_lower,
                        'bootstrap_ci_upper': ci_upper
                    }
                    
                except Exception as e:
                    logger.error(f"Failed to calculate correlation for {metric}: {e}")
                    continue
        
        return correlation_results
    
    def perform_agreement_analysis(self, deepseek_df: pd.DataFrame, 
                                 human_df: pd.DataFrame = None) -> Dict:
        """
        Perform inter-rater agreement analysis between DeepSeek and human evaluators
        
        Args:
            deepseek_df: DataFrame with DeepSeek evaluations
            human_df: DataFrame with human evaluations (if available)
            
        Returns:
            Dictionary with agreement statistics
        """
        agreement_results = {}
        
        if human_df is not None:
            # Merge on user story ID
            merged_df = pd.merge(deepseek_df, human_df, on='user_story_id', how='inner')
            
            if len(merged_df) > 0:
                # Map categories to numerical values for agreement analysis
                deepseek_categories = merged_df['human_equivalent_category']
                human_categories = merged_df['human_category']  # Assuming this column exists
                
                # Cohen's Kappa
                kappa = cohen_kappa_score(human_categories, deepseek_categories)
                
                # Confusion matrix
                cm = confusion_matrix(human_categories, deepseek_categories)
                
                # Percentage agreement
                exact_agreement = (human_categories == deepseek_categories).mean()
                
                agreement_results = {
                    'cohens_kappa': kappa,
                    'exact_agreement_percentage': exact_agreement * 100,
                    'confusion_matrix': cm.tolist(),
                    'categories': sorted(set(human_categories) | set(deepseek_categories)),
                    'sample_size': len(merged_df)
                }
        
        # Self-consistency analysis for DeepSeek
        rating_column = 'LLM Evaluator Rating' if 'LLM Evaluator Rating' in deepseek_df.columns else 'deepseek_rating'
        deepseek_ratings = deepseek_df[rating_column]
        valid_ratings = deepseek_ratings[deepseek_ratings > 0]  # Exclude error cases
        
        if len(valid_ratings) > 0:
            # Bootstrap analysis of rating consistency
            bootstrap_result = self.bootstrap.bootstrap_metric(valid_ratings.values)
            
            agreement_results['deepseek_consistency'] = {
                'mean_rating': bootstrap_result.mean,
                'std_rating': bootstrap_result.std,
                'confidence_interval': bootstrap_result.confidence_interval,
                'rating_distribution': {
                    'rating_5': (valid_ratings == 5).sum(),
                    'rating_4': (valid_ratings == 4).sum(),
                    'rating_3': (valid_ratings == 3).sum(),
                    'rating_2': (valid_ratings == 2).sum(),
                    'rating_1': (valid_ratings == 1).sum()
                },
                'rating_percentages': {
                    'excellent_percent': (valid_ratings == 5).mean() * 100,
                    'good_percent': (valid_ratings == 4).mean() * 100,
                    'moderate_percent': (valid_ratings == 3).mean() * 100,
                    'below_average_percent': (valid_ratings == 2).mean() * 100,
                    'poor_percent': (valid_ratings == 1).mean() * 100
                }
            }
        
        return agreement_results
    
    def generate_research_report(self, 
                               deepseek_results_df: pd.DataFrame,
                               correlation_results: Dict,
                               agreement_results: Dict,
                               output_path: str = "deepseek_research_report.json") -> Dict:
        """
        Generate comprehensive research report
        
        Args:
            deepseek_results_df: DeepSeek evaluation results
            correlation_results: Correlation analysis results
            agreement_results: Agreement analysis results
            output_path: Path to save the report
            
        Returns:
            Complete research report dictionary
        """
        # Map to human categories
        mapped_df = self.map_deepseek_to_human_categories(deepseek_results_df)
        
        # Handle both column name formats
        rating_column = 'LLM Evaluator Rating' if 'LLM Evaluator Rating' in mapped_df.columns else 'deepseek_rating'
        
        # Calculate summary statistics
        total_evaluations = len(mapped_df)
        successful_evaluations = len(mapped_df[mapped_df[rating_column] > 0])
        error_rate = (total_evaluations - successful_evaluations) / total_evaluations * 100
        
        # Rating distribution analysis
        rating_distribution = mapped_df[rating_column].value_counts().sort_index()
        category_distribution = mapped_df['human_equivalent_category'].value_counts()
        
        # Compile comprehensive report
        report = {
            'evaluation_overview': {
                'total_scenarios_evaluated': total_evaluations,
                'successful_evaluations': successful_evaluations,
                'error_rate_percentage': error_rate,
                'evaluation_model': 'deepseek-chat',
                'evaluation_timestamp': pd.Timestamp.now().isoformat()
            },
            'rating_distribution': {
                'raw_deepseek_ratings': rating_distribution.to_dict(),
                'mapped_human_categories': category_distribution.to_dict(),
                'percentages': {
                    'excellent_instrumental': (mapped_df['human_equivalent_category'] == 'Instrumental').mean() * 100,
                    'highly_helpful': (mapped_df['human_equivalent_category'] == 'Highly Helpful').mean() * 100,
                    'moderate_helpful': (mapped_df['human_equivalent_category'] == 'Helpful').mean() * 100,
                    'somewhat_helpful': (mapped_df['human_equivalent_category'] == 'Somewhat Helpful').mean() * 100,
                    'poor_misleading': (mapped_df['human_equivalent_category'] == 'Misleading').mean() * 100,
                    'unavailable_error': (mapped_df['human_equivalent_category'] == 'Unavailable').mean() * 100
                }
            },
            'correlation_with_automated_metrics': correlation_results,
            'agreement_analysis': agreement_results,
            'statistical_significance': {
                'significant_correlations': [
                    metric for metric, results in correlation_results.items()
                    if results.get('spearman_significant', False)
                ],
                'strong_correlations': [
                    metric for metric, results in correlation_results.items()
                    if abs(results.get('spearman_correlation', 0)) > 0.5 and 
                       results.get('spearman_significant', False)
                ]
            },
            'research_implications': self._generate_research_implications(
                correlation_results, agreement_results, mapped_df, rating_column
            )
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Research report saved to {output_path}")
        return report
    
    def _generate_research_implications(self, correlation_results: Dict, 
                                      agreement_results: Dict, 
                                      mapped_df: pd.DataFrame,
                                      rating_column: str) -> Dict:
        """Generate research implications section"""
        implications = {
            'llm_evaluation_effectiveness': {},
            'correlation_insights': {},
            'methodological_recommendations': []
        }
        
        # Analyze LLM evaluation effectiveness
        successful_rate = (mapped_df[rating_column] > 0).mean() * 100
        excellent_rate = (mapped_df['human_equivalent_category'] == 'Instrumental').mean() * 100
        
        implications['llm_evaluation_effectiveness'] = {
            'successful_evaluation_rate': successful_rate,
            'excellent_scenario_identification_rate': excellent_rate,
            'effectiveness_assessment': 'High' if successful_rate > 90 else 'Moderate' if successful_rate > 70 else 'Low'
        }
        
        # Analyze correlation insights
        strong_correlations = []
        for metric, results in correlation_results.items():
            if abs(results.get('spearman_correlation', 0)) > 0.5 and results.get('spearman_significant', False):
                strong_correlations.append({
                    'metric': metric,
                    'correlation': results['spearman_correlation'],
                    'interpretation': 'Strong positive' if results['spearman_correlation'] > 0.5 else 'Strong negative'
                })
        
        implications['correlation_insights'] = {
            'strong_correlations_found': len(strong_correlations),
            'best_correlated_metrics': strong_correlations,
            'validation_status': 'Validated' if strong_correlations else 'Requires further investigation'
        }
        
        # Methodological recommendations
        if successful_rate > 85:
            implications['methodological_recommendations'].append(
                "DeepSeek evaluation shows high reliability and can be used as a complement to human evaluation"
            )
        
        if strong_correlations:
            implications['methodological_recommendations'].append(
                f"Strong correlations with {len(strong_correlations)} automated metrics suggest LLM evaluation aligns with traditional metrics"
            )
        
        if excellent_rate > 30:
            implications['methodological_recommendations'].append(
                "High rate of excellent scenario identification suggests effective quality discrimination"
            )
        
        return implications