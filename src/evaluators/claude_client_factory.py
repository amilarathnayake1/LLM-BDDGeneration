import os
import logging
from anthropic import Anthropic

logger = logging.getLogger(__name__)

class ClaudeClientFactory:
    """
    Factory for creating Claude API clients.
    Supports both direct Anthropic API and AWS Bedrock based on environment variables.
    """
    
    @staticmethod
    def create_client(api_key: str = None):
        """
        Create appropriate Claude client based on environment configuration.
        
        Environment variables:
        - CLAUDE_PROVIDER: 'anthropic' (default) or 'bedrock'
        - AWS_REGION: AWS region for Bedrock (default: 'us-east-1')
        - AWS_ACCESS_KEY_ID: AWS access key (required for Bedrock)
        - AWS_SECRET_ACCESS_KEY: AWS secret key (required for Bedrock)
        - AWS_SESSION_TOKEN: Session token (required for temporary credentials)
        
        Returns:
            Anthropic client instance (or AnthropicBedrock client)
        """
        provider = os.getenv('CLAUDE_PROVIDER', 'anthropic').lower()
        
        if provider == 'bedrock':
            return ClaudeClientFactory._create_bedrock_client()
        else:
            return ClaudeClientFactory._create_anthropic_client(api_key)
    
    @staticmethod
    def _create_anthropic_client(api_key: str):
        """Create standard Anthropic API client"""
        logger.info("Initializing Anthropic API client")
        return Anthropic(api_key=api_key)
    
    @staticmethod
    def _create_bedrock_client():
        """Create AWS Bedrock client for Claude"""
        logger.info("Initializing AWS Bedrock client")
        
        try:
            from anthropic import AnthropicBedrock
        except ImportError:
            raise ImportError(
                "AWS Bedrock support requires the anthropic library with Bedrock support. "
                "Install with: pip install anthropic[bedrock]"
            )
        
        aws_region = os.getenv('AWS_REGION', 'us-east-1')
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_session_token = os.getenv('AWS_SESSION_TOKEN')
        
        # Check if we have credentials
        if not aws_access_key or not aws_secret_key:
            raise ValueError(
                "AWS credentials not found. Please set AWS_ACCESS_KEY_ID and "
                "AWS_SECRET_ACCESS_KEY environment variables."
            )
        
        # Check if using temporary credentials (starts with ASIA)
        if aws_access_key.startswith('ASIA'):
            if not aws_session_token:
                raise ValueError(
                    "Your AWS Access Key starts with 'ASIA', which indicates temporary credentials. "
                    "Temporary credentials require AWS_SESSION_TOKEN. Please set it in your .env file. "
                    "\n\nIf you're using AWS Academy/Educate, copy all three values from the console:\n"
                    "  - AWS_ACCESS_KEY_ID\n"
                    "  - AWS_SECRET_ACCESS_KEY\n"
                    "  - AWS_SESSION_TOKEN\n"
                    "\nOr ask your administrator for permanent credentials (starting with AKIA)."
                )
            logger.info(f"Using AWS Bedrock with temporary credentials (session token) in region: {aws_region}")
            os.environ['AWS_SESSION_TOKEN'] = aws_session_token
        else:
            logger.info(f"Using AWS Bedrock with permanent credentials in region: {aws_region}")
        
        # Set credentials as environment variables for boto3
        os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_key
        os.environ['AWS_DEFAULT_REGION'] = aws_region
        
        # Create Bedrock client - it will use boto3's credential chain
        return AnthropicBedrock(
            aws_region=aws_region
        )
    
    @staticmethod
    def get_model_name(model_name: str = None) -> str:
        """
        Get the appropriate model name based on provider.
        
        Args:
            model_name: Original model name (Anthropic format)
            
        Returns:
            Model name in the correct format for the current provider
        """
        provider = os.getenv('CLAUDE_PROVIDER', 'anthropic').lower()
        
        if provider == 'bedrock':
            # Convert Anthropic model names to Bedrock format
            # NOTE: Claude Sonnet 4 is NOT yet available in Bedrock
            # Use Claude 3.5 Sonnet v2 as the default for Bedrock
            bedrock_models = {
                # Latest available models in Bedrock
                'claude-3-5-sonnet-20241022': 'us.anthropic.claude-3-5-sonnet-20241022-v2:0',
                'claude-3-5-sonnet-20240620': 'anthropic.claude-3-5-sonnet-20240620-v1:0',
                'claude-3-opus-20240229': 'anthropic.claude-3-opus-20240229-v1:0',
                'claude-3-sonnet-20240229': 'anthropic.claude-3-sonnet-20240229-v1:0',
                'claude-3-haiku-20240307': 'anthropic.claude-3-haiku-20240307-v1:0',
                
                # Map Claude Sonnet 4 to the closest available model (Claude 3.5 Sonnet v2)
                'claude-sonnet-4-20250514': 'us.anthropic.claude-3-5-sonnet-20241022-v2:0',
            }
            
            if model_name in bedrock_models:
                bedrock_model = bedrock_models[model_name]
                if model_name == 'claude-sonnet-4-20250514':
                    logger.warning(
                        f"Claude Sonnet 4 is not yet available in Bedrock. "
                        f"Using Claude 3.5 Sonnet v2 instead: {bedrock_model}"
                    )
                else:
                    logger.info(f"Converting model name for Bedrock: {model_name} -> {bedrock_model}")
                return bedrock_model
            elif model_name and (model_name.startswith('anthropic.') or model_name.startswith('us.anthropic.')):
                # Already in Bedrock format
                return model_name
            else:
                # Default to Claude 3.5 Sonnet v2 for Bedrock (best available model)
                default_model = 'us.anthropic.claude-3-5-sonnet-20241022-v2:0'
                logger.warning(
                    f"Unknown model '{model_name}', using Claude 3.5 Sonnet v2 (best available in Bedrock): "
                    f"{default_model}"
                )
                return default_model
        else:
            # Return as-is for Anthropic API
            return model_name or 'claude-sonnet-4-20250514'
