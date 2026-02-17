import pandas as pd
import json
import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import os
import boto3
from botocore.exceptions import ClientError
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
    """Configuration for the DeepSeek Bedrock evaluator"""
    model_name: str = "us.deepseek.r1-v1:0"  # DeepSeek-R1 in Bedrock
    aws_region: str = "us-east-1"
    max_retries: int = 3
    temperature: float = 0.0
    max_tokens: int = 2000
    batch_size: int = 5
    delay_between_requests: float = 1.0

class DeepSeekBedrockBDDEvaluator:
    """
    DeepSeek BDD Evaluator using AWS Bedrock
    
    Supports DeepSeek-R1 and DeepSeek-V3.1 models in Bedrock
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.client = None
        self._setup_client()
        
        self.evaluation_prompt = """You are an expert software testing evaluator specialising in Behaviour-Driven Development (BDD) scenarios. Your task is to evaluate the BDD scenario against the provided requirements (both user story and description) and rate it on a scale from 1 to 5, where a higher score indicates better quality.

**Evaluation Context:**

**USER STORY:**
{user_story}

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
}}

IMPORTANT: You must respond ONLY with valid JSON. No additional text before or after the JSON."""
    
    def _setup_client(self):
        """Initialize the AWS Bedrock client"""
        try:
            # Get AWS credentials from environment
            aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            aws_session_token = os.getenv('AWS_SESSION_TOKEN')  # For temporary credentials
            
            # Create session with credentials if available
            if aws_access_key_id and aws_secret_access_key:
                session = boto3.Session(
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    aws_session_token=aws_session_token,
                    region_name=self.config.aws_region
                )
                self.client = session.client('bedrock-runtime')
            else:
                # Use default credentials (IAM role, AWS CLI config, etc.)
                self.client = boto3.client(
                    'bedrock-runtime',
                    region_name=self.config.aws_region
                )
            
            logger.info(f"AWS Bedrock client initialized successfully for region: {self.config.aws_region}")
            logger.info(f"Using DeepSeek model: {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize AWS Bedrock client: {e}")
            raise
    
    @backoff.on_exception(
        backoff.expo,
        (ClientError,),
        max_tries=3,
        giveup=lambda e: e.response['Error']['Code'] not in ['ThrottlingException', 'ModelTimeoutException']
    )
    def _make_api_call(self, prompt: str) -> Dict:
        """
        Make API call to DeepSeek via AWS Bedrock with retry logic
        
        Args:
            prompt: The evaluation prompt
            
        Returns:
            Dictionary containing rating and reasoning
        """
        try:
            # Prepare the request body for DeepSeek-R1 in Bedrock
            # Use Converse API for better compatibility
            messages = [
                {
                    "role": "user",
                    "content": [{"text": prompt}]
                }
            ]
            
            # System prompt
            system_prompts = [
                {
                    "text": "You are an expert BDD scenario evaluator. Always respond with valid JSON only."
                }
            ]
            
            # Inference configuration
            inference_config = {
                "temperature": self.config.temperature,
                "maxTokens": self.config.max_tokens
            }
            
            # Make the API call using Converse API
            response = self.client.converse(
                modelId=self.config.model_name,
                messages=messages,
                system=system_prompts,
                inferenceConfig=inference_config
            )
            
            # Extract the response content
            # DeepSeek-R1 returns reasoning in 'reasoningContent' field
            response_text = ""
            content = response['output']['message']['content']
            
            for item in content:
                # DeepSeek-R1 uses reasoningContent for its thinking process
                if 'reasoningContent' in item:
                    reasoning = item['reasoningContent']
                    if 'reasoningText' in reasoning and 'text' in reasoning['reasoningText']:
                        response_text += reasoning['reasoningText']['text']
                # Some models also have direct text
                elif 'text' in item:
                    response_text += item['text']
            
            # Log token usage
            token_usage = response.get('usage', {})
            logger.debug(f"Token usage - Input: {token_usage.get('inputTokens', 0)}, "
                        f"Output: {token_usage.get('outputTokens', 0)}")
            
            # Verify we got a response
            if not response_text:
                raise ValueError("No text content in response")
            
            # Parse JSON response
            # DeepSeek-R1 includes both JSON and reasoning text
            # We need to extract just the JSON part
            response_text = response_text.strip()
            
            # Try to find JSON in the response
            # Method 1: Look for markdown code blocks
            if '```json' in response_text:
                start = response_text.find('```json') + 7
                end = response_text.find('```', start)
                if end > start:
                    response_text = response_text[start:end].strip()
            elif '```' in response_text:
                start = response_text.find('```') + 3
                end = response_text.find('```', start)
                if end > start:
                    response_text = response_text[start:end].strip()
            
            # Method 2: Look for JSON object boundaries
            # Find the first { and last } to extract just the JSON
            if '{' in response_text and '}' in response_text:
                start = response_text.find('{')
                # Find the matching closing brace
                brace_count = 0
                end = start
                for i in range(start, len(response_text)):
                    if response_text[i] == '{':
                        brace_count += 1
                    elif response_text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = i + 1
                            break
                
                if end > start:
                    response_text = response_text[start:end].strip()
            
            # Clean up any remaining artifacts
            response_text = response_text.strip()
            
            result = json.loads(response_text)
            
            # Validate response structure
            if not all(key in result for key in ['rating', 'reasoning']):
                raise ValueError("Invalid response structure from DeepSeek")
            
            if result['rating'] not in [1, 2, 3, 4, 5]:
                raise ValueError(f"Invalid rating: {result['rating']}")
                
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse DeepSeek response as JSON: {e}")
            logger.error(f"Response text: {response_text}")
            raise
        except ClientError as e:
            error_code = e.response['Error']['Code']
            logger.error(f"AWS Bedrock API error ({error_code}): {e}")
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
        
        # Filter data to resume from specific point
        data_to_evaluate = data_df.iloc[resume_from:].copy()
        logger.info(f"Data to evaluate after filtering: {len(data_to_evaluate)} rows")
        
        if len(data_to_evaluate) == 0:
            logger.warning("No data to evaluate after filtering!")
            return self._results_to_dataframe(results)
        
        for idx, row in data_to_evaluate.iterrows():
            try:
                logger.info(f"Evaluating scenario {idx + 1}/{total_scenarios}")
                
                result = self.evaluate_single_scenario(
                    user_story=str(row['User Story']),
                    description=str(row['Description']),
                    reference_scenario=str(row['BDD-Reference']),
                    test_scenario=str(row['BDD-AI Generated']),
                    user_story_id=row['#']
                )
                
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
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        logger.info(f"Collected {len(results)} evaluation results")
        
        # Convert results to DataFrame
        results_df = self._results_to_dataframe(results)
        
        # Save final results
        if output_file:
            self._save_results(results_df, output_file)
            
            # Also save detailed version with reasoning
            detailed_df = self._results_to_detailed_dataframe(results)
            
            import datetime
            now = datetime.datetime.now()
            timestamp_str = now.strftime('%Y-%m-%d_%H-%M-%S')
            
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
            results_df: DataFrame with evaluation results
            
        Returns:
            Dictionary with statistical measures
        """
        if 'LLM Evaluator Rating' not in results_df.columns:
            available_columns = list(results_df.columns)
            logger.error(f"Expected 'LLM Evaluator Rating' column not found. Available columns: {available_columns}")
            return {"error": f"Rating column not found. Available columns: {available_columns}"}
        
        # Filter out error cases
        valid_results = results_df[results_df['LLM Evaluator Rating'] > 0]
        
        if len(valid_results) == 0:
            return {"error": "No valid evaluations found"}
        
        ratings = valid_results['LLM Evaluator Rating']
        
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
