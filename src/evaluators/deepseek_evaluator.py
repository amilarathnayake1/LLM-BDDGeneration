import pandas as pd
import openai
import json
import time
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import backoff

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Data class for storing evaluation results"""
    user_story_id: int
    user_story: str
    description: str
    reference_scenario: str
    ai_generated_scenario: str
    deepseek_rating: int
    deepseek_reasoning: str
    evaluation_timestamp: str

@dataclass
class EvaluationConfig:
    """Configuration for the DeepSeek evaluator"""
    model_name: str = "deepseek-chat"
    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    max_retries: int = 3
    temperature: float = 0.0
    max_tokens: int = 1000
    batch_size: int = 5
    delay_between_requests: float = 1.0

class DeepSeekBDDEvaluator:
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.client = None
        self._setup_client()
        
        self.evaluation_prompt = """ You are an expert software testing evaluator specialising in Behaviour-Driven Development (BDD) scenarios. Your task is to evaluate the BDD scenario against the provided requirements (both user story and description) and rate it on a scale from 1 to 5, where a higher score indicates better quality.
**Evaluation Context:*

**USER STORY:**
* {user_story}

**DESCRIPTION:**
{description}

**BDD SCENARIO:**
{ai_generated_scenario}

**Evaluation Instructions:**
When evaluating the BDD scenario, consider ALL provided requirements.
A good BDD scenario should have:
- Clear, specific, and testable steps
- Comprehensive coverage of ALL provided requirements
- Excellent clarity and precision in test assertions
- Proper Gherkin syntax and BDD structure

**Response Format:**
Please respond in the following JSON format:
{{
    "rating": <1, 2, 3, 4, or 5>,
    "reasoning": "Detailed explanation of your evaluation, including specific strengths and weaknesses"
}}"""
    
    def _setup_client(self):
        """Initialize the OpenAI client for DeepSeek API"""
        try:
            self.client = openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url
            )
            logger.info("DeepSeek client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DeepSeek client: {e}")
            raise
    
    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, openai.APITimeoutError),
        max_tries=3
    )
    def _make_api_call(self, prompt: str) -> Dict:
        """
        Make API call to DeepSeek with retry logic
        
        Args:
            prompt: The evaluation prompt
            
        Returns:
            Dictionary containing rating and reasoning
        """
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert BDD scenario evaluator. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # Validate response structure
            if not all(key in result for key in ['rating', 'reasoning']):
                raise ValueError("Invalid response structure from DeepSeek")
            
            if result['rating'] not in [1, 2, 3, 4, 5]:
                raise ValueError(f"Invalid rating: {result['rating']}")
                
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse DeepSeek response as JSON: {e}")
            raise
        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise
    
    def evaluate_single_scenario(self, 
                                user_story: str,
                                description: str,
                                reference_scenario: str,
                                test_scenario: str,
                                user_story_id: int = None) -> EvaluationResult:
        """
        Evaluate a single BDD scenario pair
        
        Args:
            user_story: The user story text
            description: Detailed description/requirements
            reference_scenario: Ground truth BDD scenario
            test_scenario: BDD scenario to evaluate
            user_story_id: Optional ID for tracking
            
        Returns:
            EvaluationResult object
        """
        prompt = self.evaluation_prompt.format(
            user_story=user_story,
            description=description,
            reference_scenario=reference_scenario,
            ai_generated_scenario=test_scenario
        )
        
        try:
            result = self._make_api_call(prompt)
            
            return EvaluationResult(
                user_story_id=user_story_id,
                user_story=user_story,
                description=description,
                reference_scenario=reference_scenario,
                ai_generated_scenario=test_scenario,
                deepseek_rating=result['rating'],
                deepseek_reasoning=result['reasoning'],
                evaluation_timestamp=pd.Timestamp.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Evaluation failed for user story {user_story_id}: {e}")
            # Return error result
            return EvaluationResult(
                user_story_id=user_story_id,
                user_story=user_story,
                description=description,
                reference_scenario=reference_scenario,
                ai_generated_scenario=test_scenario,
                deepseek_rating=-1,  # Error indicator
                deepseek_reasoning=f"Evaluation failed: {str(e)}",
                evaluation_timestamp=pd.Timestamp.now().isoformat()
            )
    
    def evaluate_batch(self, data_df: pd.DataFrame, 
                      output_file: str = None,
                      resume_from: int = 0) -> pd.DataFrame:
        """
        Evaluate multiple BDD scenarios in batch
        
        Args:
            data_df: DataFrame with columns: '#', 'User Story', 'Description', 
                    'BDD-Reference', 'BDD-AI Generated'
            output_file: Optional file to save results incrementally
            resume_from: Index to resume evaluation from
            
        Returns:
            DataFrame with evaluation results
        """
        results = []
        total_scenarios = len(data_df)
        
        logger.info(f"Starting batch evaluation of {total_scenarios} scenarios")
        logger.info(f"Resuming from index: {resume_from}")
        
        # Debug: Check input data
        logger.info(f"Input DataFrame shape: {data_df.shape}")
        logger.info(f"Input DataFrame columns: {list(data_df.columns)}")
        
        # Filter data to resume from specific point
        data_to_evaluate = data_df.iloc[resume_from:].copy()
        logger.info(f"Data to evaluate after filtering: {len(data_to_evaluate)} rows")
        
        if len(data_to_evaluate) == 0:
            logger.warning("No data to evaluate after filtering!")
            return self._results_to_dataframe(results)
        
        for idx, row in data_to_evaluate.iterrows():
            try:
                logger.info(f"Evaluating scenario {idx + 1}/{total_scenarios}")
                
                # Debug: Check row data
                logger.debug(f"Row data: {row.to_dict()}")
                
                result = self.evaluate_single_scenario(
                    user_story=str(row['User Story']),
                    description=str(row['Description']),
                    reference_scenario=str(row['BDD-Reference']),
                    test_scenario=str(row['BDD-AI Generated']),
                    user_story_id=row['#']
                )
                
                # Debug: Check result
                logger.debug(f"Evaluation result: rating={result.deepseek_rating}")
                
                results.append(result)
                
                # Save incrementally if output file specified
                if output_file and len(results) % 5 == 0:
                    self._save_incremental_results(results, output_file, resume_from)
                
                # Rate limiting
                time.sleep(self.config.delay_between_requests)
                
            except KeyboardInterrupt:
                logger.info("Evaluation interrupted by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error at index {idx}: {e}")
                logger.error(f"Row data: {row.to_dict()}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        logger.info(f"Collected {len(results)} evaluation results")
        
        # Convert results to DataFrame (user-specified format)
        results_df = self._results_to_dataframe(results)
        logger.info(f"Results DataFrame shape: {results_df.shape}")
        logger.info(f"Results DataFrame columns: {list(results_df.columns)}")
        
        # Save final results
        if output_file:
            self._save_results(results_df, output_file)
            
            # Also save detailed version with reasoning (add timestamp to filename)
            detailed_df = self._results_to_detailed_dataframe(results)
            
            # Generate timestamped filename for detailed results
            import datetime
            now = datetime.datetime.now()
            timestamp_str = now.strftime('%Y-%m-%d_%H-%M-%S')
            
            # Insert timestamp into filename
            if output_file.endswith('.xlsx'):
                detailed_output_file = output_file.replace('.xlsx', f'_detailed_{timestamp_str}.xlsx')
            elif output_file.endswith('.csv'):
                detailed_output_file = output_file.replace('.csv', f'_detailed_{timestamp_str}.csv')
            else:
                detailed_output_file = f"{output_file}_detailed_{timestamp_str}"
            
            self._save_results(detailed_df, detailed_output_file)
            logger.info(f"Detailed results with reasoning saved to {detailed_output_file}")
            
        logger.info(f"Batch evaluation completed. Evaluated {len(results)} scenarios")
        return results_df
    
    def _results_to_dataframe(self, results: List[EvaluationResult]) -> pd.DataFrame:
        """Convert evaluation results to DataFrame with user-specified format"""
        data = []
        for result in results:
            data.append({
                '#': result.user_story_id,
                'User Story': result.user_story,
                'Description': result.description,
                'BDD-Reference': result.reference_scenario,
                'BDD-AI Generated': result.ai_generated_scenario,
                'LLM Evaluator Rating': result.deepseek_rating
            })
        return pd.DataFrame(data)
    
    def _results_to_detailed_dataframe(self, results: List[EvaluationResult]) -> pd.DataFrame:
        """Convert evaluation results to detailed DataFrame with additional info"""
        data = []
        for result in results:
            data.append({
                '#': result.user_story_id,
                'User Story': result.user_story,
                'Description': result.description,
                'BDD-Reference': result.reference_scenario,
                'BDD-AI Generated': result.ai_generated_scenario,
                'LLM Evaluator Rating': result.deepseek_rating,
                'LLM Evaluator Reasoning': result.deepseek_reasoning,
                'Evaluation Timestamp': result.evaluation_timestamp
            })
        return pd.DataFrame(data)
    
    def _save_results(self, results_df: pd.DataFrame, output_file: str):
        """Save results to file"""
        try:
            if output_file.endswith('.xlsx'):
                results_df.to_excel(output_file, index=False)
            elif output_file.endswith('.csv'):
                results_df.to_csv(output_file, index=False)
            else:
                results_df.to_json(output_file, orient='records', indent=2)
            
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def _save_incremental_results(self, results: List[EvaluationResult], 
                                 output_file: str, resume_from: int):
        """Save incremental results for recovery"""
        backup_file = f"{output_file.rsplit('.', 1)[0]}_incremental_{resume_from}.json"
        results_df = self._results_to_dataframe(results)
        results_df.to_json(backup_file, orient='records', indent=2)
        logger.info(f"Incremental backup saved to {backup_file}")
    
    def calculate_statistics(self, results_df: pd.DataFrame) -> Dict:
        """
        Calculate evaluation statistics
        
        Args:
            results_df: DataFrame with evaluation results (either format)
            
        Returns:
            Dictionary with statistical measures
        """
        # Handle both column name formats
        if 'LLM Evaluator Rating' in results_df.columns:
            rating_column = 'LLM Evaluator Rating'
        else:
            # Check what columns are actually available
            available_columns = list(results_df.columns)
            logger.error(f"Expected 'LLM Evaluator Rating' column not found. Available columns: {available_columns}")
            return {"error": f"Rating column not found. Available columns: {available_columns}"}
        
        # Filter out error cases
        valid_results = results_df[results_df[rating_column] > 0]
        
        if len(valid_results) == 0:
            return {"error": "No valid evaluations found"}
        
        ratings = valid_results[rating_column]
        
        statistics = {
            'total_evaluated': len(valid_results),
            'total_errors': len(results_df) - len(valid_results),
            'rating_distribution': {
                'rating_5_excellent': len(ratings[ratings == 5]),
                'rating_4_good': len(ratings[ratings == 4]),
                'rating_3_moderate': len(ratings[ratings == 3]),
                'rating_2_below_average': len(ratings[ratings == 2]),
                'rating_1_poor': len(ratings[ratings == 1])
            },
            'rating_percentages': {
                'rating_5_percent': (len(ratings[ratings == 5]) / len(ratings)) * 100,
                'rating_4_percent': (len(ratings[ratings == 4]) / len(ratings)) * 100,
                'rating_3_percent': (len(ratings[ratings == 3]) / len(ratings)) * 100,
                'rating_2_percent': (len(ratings[ratings == 2]) / len(ratings)) * 100,
                'rating_1_percent': (len(ratings[ratings == 1]) / len(ratings)) * 100
            },
            'mean_rating': float(ratings.mean()),
            'median_rating': float(ratings.median()),
            'std_rating': float(ratings.std()),
            'rating_mode': int(ratings.mode().iloc[0]) if len(ratings.mode()) > 0 else None
        }
        
        return statistics

# Utility functions
def load_excel_data(file_path: str, column_mapping: Dict[str, str] = None) -> pd.DataFrame:
    """Load BDD scenario data from Excel file with flexible column mapping"""
    try:
        df = pd.read_excel(file_path)
        
        # If no mapping provided, use default column names
        if column_mapping is None:
            required_columns = ['#', 'User Story', 'Description', 'BDD-Reference', 'BDD-AI Generated']
            
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Excel file must contain columns: {required_columns}")
        else:
            # Rename columns based on mapping
            reverse_mapping = {v: k for k, v in column_mapping.items()}
            df = df.rename(columns=reverse_mapping)
            
            # Verify all required columns are now present
            required_columns = ['#', 'User Story', 'Description', 'BDD-Reference', 'BDD-AI Generated']
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                raise ValueError(f"After mapping, still missing columns: {missing}")
        
        logger.info(f"Loaded {len(df)} scenarios from {file_path}")
        return df
    
    except Exception as e:
        logger.error(f"Failed to load Excel data: {e}")
        raise