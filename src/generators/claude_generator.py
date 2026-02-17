import logging
from typing import List
from ..models import RequirementData
from config import ClaudeConfig
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClaudeGenerator:
    """
    BDD scenario generator using Claude (Anthropic API or AWS Bedrock).
    
    Supports both direct Anthropic API and AWS Bedrock based on environment variables.
    Set CLAUDE_PROVIDER='bedrock' to use AWS Bedrock, or 'anthropic' (default) for Anthropic API.
    """
    
    def __init__(self, config: ClaudeConfig, prompt_strategy: str = "zero_shot"):
        """Initialize the BDD generator with Claude configuration"""
        self.config = config
        self.prompt_strategy = prompt_strategy
        self.client = None
        self._setup_client()
        # Update model name for Bedrock if needed
        self._update_model_name()
        # Load prompt template
        self._load_prompt_template()
    
    def _setup_client(self):
        """Initialize the Claude client (Anthropic API or AWS Bedrock)"""
        try:
            from ..claude_client_factory import ClaudeClientFactory
            
            self.client = ClaudeClientFactory.create_client(
                api_key=self.config.api_key
            )
            
            provider = os.getenv('CLAUDE_PROVIDER', 'anthropic')
            logger.info(f"Claude Generator initialized successfully (provider: {provider})")
        except Exception as e:
            logger.error(f"Failed to initialize Claude client: {e}")
            raise
    
    def _update_model_name(self):
        """Update model name if using Bedrock"""
        from ..claude_client_factory import ClaudeClientFactory
        original_model = self.config.model
        self.config.model = ClaudeClientFactory.get_model_name(self.config.model)
        
        if original_model != self.config.model:
            logger.info(f"Model name updated: {original_model} -> {self.config.model}")
    
    def _load_prompt_template(self):
        """Load prompt template from file"""
        try:
            from ..prompt_loader import PromptLoader
            loader = PromptLoader()
            self.prompt_template = loader.load_prompt(self.prompt_strategy)
            logger.info(f"Loaded prompt strategy: {self.prompt_strategy}")
        except FileNotFoundError as e:
            logger.error(f"Failed to load prompt: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading prompt: {e}")
            raise

    def generate_scenario(self, requirement: RequirementData) -> str:
        """Generate BDD scenario using Claude API based on Requirements"""
        try:
            prompt = self._create_prompt(requirement)
            
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Error generating BDD scenario with Claude: {e}")
            raise Exception(f"Error generating BDD scenario with Claude: {e}")

    def _create_prompt(self, requirement: RequirementData) -> str:
        """Create the prompt for scenario generation using loaded template"""
        return self.prompt_template.format(
            user_story=requirement.user_story,
            description=requirement.requirements
        )
    
    def _create_prompt_user_story_only(self, requirement: RequirementData) -> str:
        """Create the prompt for scenario generation using ONLY user story (no description)"""
        # Replace {description} placeholder with empty string or remove it from template
        modified_template = self.prompt_template.replace(
            "Description: {description}\n", ""
        ).replace(
            "User Story: {user_story}",
            "User Story: {user_story}"
        )
        return modified_template.format(
            user_story=requirement.user_story,
            description=""  # Keep for compatibility but won't be used
        )
    
    def _create_prompt_description_only(self, requirement: RequirementData) -> str:
        """Create the prompt for scenario generation using ONLY description (no user story)"""
        # Replace {user_story} placeholder with empty string or remove it from template
        modified_template = self.prompt_template.replace(
            "User Story: {user_story}\n", ""
        ).replace(
            "Description: {description}",
            "Description: {description}"
        )
        return modified_template.format(
            user_story="",  # Keep for compatibility but won't be used
            description=requirement.requirements
        )
    
    def generate_scenario_user_story_only(self, requirement: RequirementData) -> str:
        """Generate BDD scenario using Claude API based on User Story ONLY (no description)"""
        try:
            prompt = self._create_prompt_user_story_only(requirement)
            
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Error generating BDD scenario with Claude (User Story only): {e}")
            raise Exception(f"Error generating BDD scenario with Claude (User Story only): {e}")
    
    def generate_scenario_description_only(self, requirement: RequirementData) -> str:
        """Generate BDD scenario using Claude API based on Description ONLY (no user story)"""
        try:
            prompt = self._create_prompt_description_only(requirement)
            
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Error generating BDD scenario with Claude (Description only): {e}")
            raise Exception(f"Error generating BDD scenario with Claude (Description only): {e}")


if __name__ == "__main__":
    """Run Claude generator from command line"""
    import sys
    import argparse
    from pathlib import Path
    from datetime import datetime
    import pandas as pd
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Add parent directory to path to import config and parser
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config import Config
    from src.parser import ExcelParser
    
    parser = argparse.ArgumentParser(
        description='Generate BDD scenarios using Claude',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate using User Story + Description (default)
  python -m src.generators.claude_generator --prompt zero_shot
  
  # Generate using ONLY User Story (no description)
  python -m src.generators.claude_generator --prompt zero_shot --user-story-only
  
  # Generate using ONLY Description (no user story)
  python -m src.generators.claude_generator --prompt zero_shot --description-only
  
  # Generate for first 10 requirements only
  python -m src.generators.claude_generator --prompt zero_shot --max 10
  
  # Specify custom input/output files
  python -m src.generators.claude_generator --prompt chain_of_thought --description-only --input data/MyRequirements.xlsx --output results/my_scenarios.xlsx
        """
    )
    
    parser.add_argument(
        '--input',
        default='data/Requirements.xlsx',
        help='Input Excel file with requirements (default: data/Requirements.xlsx)'
    )
    
    parser.add_argument(
        '--output',
        help='Output Excel file for results (default: results/claude_scenarios_TIMESTAMP.xlsx)'
    )
    
    parser.add_argument(
        '--max',
        type=int,
        help='Maximum number of scenarios to generate (default: all)'
    )
    
    parser.add_argument(
        '--prompt',
        required=True,
        choices=['zero_shot', 'chain_of_thought', 'few_shot'],
        help='Prompt strategy to use (REQUIRED). Choose: zero_shot, chain_of_thought, or few_shot'
    )
    
    parser.add_argument(
        '--user-story-only',
        action='store_true',
        help='Generate BDD scenarios using ONLY User Story (excludes Description field)'
    )
    
    parser.add_argument(
        '--description-only',
        action='store_true',
        help='Generate BDD scenarios using ONLY Description (excludes User Story field)'
    )
    
    args = parser.parse_args()
    
    # Validate that user-story-only and description-only are not both set
    if args.user_story_only and args.description_only:
        parser.error("Cannot use both --user-story-only and --description-only flags together")
    
    try:
        logger.info("=" * 80)
        logger.info("BDD SCENARIO GENERATION - CLAUDE")
        logger.info("=" * 80)
        
        # Show which provider is being used
        provider = os.getenv('CLAUDE_PROVIDER', 'anthropic')
        logger.info(f"Using provider: {provider}")
        
        # Initialize configuration
        config = Config.get_default_config()
        
        # For Bedrock, use Claude 3.5 Sonnet
        if provider.lower() == 'bedrock':
            config.claude.model = "claude-3-5-sonnet-20241022"
            logger.info(f"Model: {config.claude.model} (Bedrock)")
        else:
            logger.info(f"Model: {config.claude.model} (Anthropic API)")
        
        # Create generator with specified prompt strategy
        logger.info("\nInitializing Claude generator...")
        generator = ClaudeGenerator(config.claude, prompt_strategy=args.prompt)
        logger.info("✓ Generator initialized successfully")
        
        # Log the input mode being used
        if args.user_story_only:
            logger.info("\n⚠️  INPUT MODE: User Story ONLY (Description will be excluded)")
        elif args.description_only:
            logger.info("\n⚠️  INPUT MODE: Description ONLY (User Story will be excluded)")
        else:
            logger.info("\n✓ INPUT MODE: User Story + Description (default)")
        
        # Read requirements
        logger.info(f"\nReading requirements from: {args.input}")
        requirements = ExcelParser.read_requirements(Path(args.input))
        logger.info(f"✓ Loaded {len(requirements)} requirements")
        
        # Limit number of scenarios if specified
        if args.max and args.max < len(requirements):
            requirements = requirements[:args.max]
            logger.info(f"Limiting to first {args.max} scenarios")
        
        # Generate scenarios
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING SCENARIOS")
        logger.info("=" * 80)
        
        results = []
        
        for idx, requirement in enumerate(requirements, 1):
            logger.info(f"\n[{idx}/{len(requirements)}] Generating scenario for ID: {requirement.id}")
            
            try:
                # Generate scenario - use appropriate mode based on flags
                if args.user_story_only:
                    logger.info("  Mode: User Story ONLY (excluding Description)")
                    scenario = generator.generate_scenario_user_story_only(requirement)
                elif args.description_only:
                    logger.info("  Mode: Description ONLY (excluding User Story)")
                    scenario = generator.generate_scenario_description_only(requirement)
                else:
                    logger.info("  Mode: User Story + Description")
                    scenario = generator.generate_scenario(requirement)
                
                # Store result
                results.append({
                    '#': requirement.id,
                    'User Story': requirement.user_story,
                    'Description': requirement.requirements,
                    'BDD-Reference': requirement.manual_scenario,
                    'BDD-AI Generated': scenario
                })
                
                # Show preview
                logger.info("✓ Scenario generated successfully")
                logger.info("Preview (first 150 chars):")
                logger.info(f"  {scenario[:150]}...")
                
            except Exception as e:
                logger.error(f"✗ Failed to generate scenario: {e}")
                results.append({
                    '#': requirement.id,
                    'User Story': requirement.user_story,
                    'Description': requirement.requirements,
                    'BDD-Reference': requirement.manual_scenario,
                    'BDD-AI Generated': f"ERROR: {str(e)}"
                })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Generate output filename if not provided
        if args.output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if args.user_story_only:
                mode_suffix = "_user_story_only"
            elif args.description_only:
                mode_suffix = "_description_only"
            else:
                mode_suffix = ""
            args.output = f"results/claude_scenarios_{args.prompt}{mode_suffix}_{timestamp}.xlsx"
        
        # Save results
        logger.info("\n" + "=" * 80)
        logger.info("SAVING RESULTS")
        logger.info("=" * 80)
        
        # Create results directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Save to Excel
        results_df.to_excel(args.output, index=False)
        logger.info(f"✓ Results saved to: {args.output}")
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total requirements: {len(requirements)}")
        logger.info(f"Successfully generated: {len([r for r in results if not r['BDD-AI Generated'].startswith('ERROR')])}")
        logger.info(f"Failed: {len([r for r in results if r['BDD-AI Generated'].startswith('ERROR')])}")
        logger.info(f"Output file: {args.output}")
        
        logger.info("\n✓ Generation complete!")
        
    except KeyboardInterrupt:
        logger.info("\n\n✗ Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Generation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
