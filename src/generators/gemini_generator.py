import google.generativeai as genai
import logging
from typing import List
from ..models import RequirementData
from config import GeminiConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiGenerator:
    def __init__(self, config: GeminiConfig, prompt_strategy: str = "few_shot"):
        """Initialize the BDD generator with Gemini configuration"""
        self.config = config
        self.prompt_strategy = prompt_strategy
        genai.configure(api_key=config.api_key)
        
        # Use Gemini 2.5 Flash - stable, fast, and capable
        model_name = "models/gemini-2.5-flash"
        
        try:
            self.model = genai.GenerativeModel(model_name)
            logger.info(f"Successfully initialized with model: {model_name}")
        except Exception as e:
            raise Exception(f"Failed to initialize Gemini model: {e}")
        
        self._load_prompt_template()
    
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
        """Generate BDD scenario using Gemini API based on User Story and Requirements"""
        try:
            prompt = self._create_prompt(requirement)
            
            # Configure safety settings to be less restrictive
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_tokens,
                ),
                safety_settings=safety_settings
            )
            
            # Check finish reason before accessing text
            if response.candidates:
                candidate = response.candidates[0]
                finish_reason = candidate.finish_reason
                
                # finish_reason: 1=STOP (success), 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION, 5=OTHER
                if finish_reason == 3:  # SAFETY
                    raise Exception("Response blocked by safety filters. Try simplifying the prompt.")
                elif finish_reason == 2:  # MAX_TOKENS
                    raise Exception("Response exceeded max tokens. Increase max_tokens in config.")
                elif finish_reason != 1:  # Not STOP (normal completion)
                    raise Exception(f"Generation stopped with reason: {finish_reason}")
                
                # Extract text from parts
                if candidate.content and candidate.content.parts:
                    text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
                    if text_parts:
                        generated_text = ''.join(text_parts).strip()
                        # Clean response to remove markdown formatting
                        generated_text = self._clean_response(generated_text)
                        return generated_text
            
            raise Exception("No valid response generated")
            
        except Exception as e:
            raise Exception(f"Error generating BDD scenario with Gemini: {e}")

    def _clean_response(self, text: str) -> str:
        """Remove reasoning tags, markdown formatting, and extract only the BDD scenario"""
        import re
        
        # Remove <reasoning>...</reasoning> tags and their content
        text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove any other XML-style tags that might be present
        text = re.sub(r'<[^>]+>.*?</[^>]+>', '', text, flags=re.DOTALL)
        
        # Remove markdown bold formatting (**text** or __text__)
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'__(.*?)__', r'\1', text)
        
        # Remove markdown italic formatting (*text* or _text_)
        text = re.sub(r'(?<!\*)\*(?!\*)(.*?)\*(?!\*)', r'\1', text)
        text = re.sub(r'(?<!_)_(?!_)(.*?)_(?!_)', r'\1', text)
        
        # Remove bullet points at the start of lines (*, -, +)
        text = re.sub(r'^\s*[*\-+]\s+', '', text, flags=re.MULTILINE)
        
        # Remove numbered list markers (1., 2., etc.)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # Extract BDD scenario lines
        lines = text.split('\n')
        scenario_started = False
        scenario_lines = []
        
        for line in lines:
            # Look for the start of a BDD scenario
            if line.strip().startswith('Scenario:'):
                scenario_started = True
                scenario_lines.append(line)
            elif scenario_started:
                # Continue collecting scenario lines
                if line.strip().startswith(('Given', 'When', 'Then', 'And', 'But', 'Feature:')):
                    scenario_lines.append(line)
                elif line.strip() and not any(keyword in line.lower() for keyword in ['reasoning', 'explanation', 'note:', 'output']):
                    # Include non-empty lines that aren't meta-commentary
                    scenario_lines.append(line)
        
        # If we found scenario lines, use those
        if scenario_lines:
            cleaned_text = '\n'.join(scenario_lines).strip()
        else:
            # Otherwise use the tag-cleaned version
            cleaned_text = text.strip()
        
        # Final cleanup: remove multiple blank lines
        cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)
        
        return cleaned_text

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
        """Generate BDD scenario using Gemini API based on User Story ONLY (no description)"""
        try:
            prompt = self._create_prompt_user_story_only(requirement)
            
            # Configure safety settings to be less restrictive
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_tokens,
                ),
                safety_settings=safety_settings
            )
            
            # Check finish reason before accessing text
            if response.candidates:
                candidate = response.candidates[0]
                finish_reason = candidate.finish_reason
                
                # finish_reason: 1=STOP (success), 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION, 5=OTHER
                if finish_reason == 3:  # SAFETY
                    raise Exception("Response blocked by safety filters. Try simplifying the prompt.")
                elif finish_reason == 2:  # MAX_TOKENS
                    raise Exception("Response exceeded max tokens. Increase max_tokens in config.")
                elif finish_reason != 1:  # Not STOP (normal completion)
                    raise Exception(f"Generation stopped with reason: {finish_reason}")
                
                # Extract text from parts
                if candidate.content and candidate.content.parts:
                    text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
                    if text_parts:
                        generated_text = ''.join(text_parts).strip()
                        # Clean response to remove markdown formatting
                        generated_text = self._clean_response(generated_text)
                        return generated_text
            
            raise Exception("No valid response generated")
            
        except Exception as e:
            raise Exception(f"Error generating BDD scenario with Gemini (User Story only): {e}")
    
    def generate_scenario_description_only(self, requirement: RequirementData) -> str:
        """Generate BDD scenario using Gemini API based on Description ONLY (no user story)"""
        try:
            prompt = self._create_prompt_description_only(requirement)
            
            # Configure safety settings to be less restrictive
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_tokens,
                ),
                safety_settings=safety_settings
            )
            
            # Check finish reason before accessing text
            if response.candidates:
                candidate = response.candidates[0]
                finish_reason = candidate.finish_reason
                
                # finish_reason: 1=STOP (success), 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION, 5=OTHER
                if finish_reason == 3:  # SAFETY
                    raise Exception("Response blocked by safety filters. Try simplifying the prompt.")
                elif finish_reason == 2:  # MAX_TOKENS
                    raise Exception("Response exceeded max tokens. Increase max_tokens in config.")
                elif finish_reason != 1:  # Not STOP (normal completion)
                    raise Exception(f"Generation stopped with reason: {finish_reason}")
                
                # Extract text from parts
                if candidate.content and candidate.content.parts:
                    text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
                    if text_parts:
                        generated_text = ''.join(text_parts).strip()
                        # Clean response to remove markdown formatting
                        generated_text = self._clean_response(generated_text)
                        return generated_text
            
            raise Exception("No valid response generated")
            
        except Exception as e:
            raise Exception(f"Error generating BDD scenario with Gemini (Description only): {e}")


if __name__ == "__main__":
    """Run Gemini generator from command line"""
    import sys
    import argparse
    from pathlib import Path
    from datetime import datetime
    import pandas as pd
    from dotenv import load_dotenv
    import os
    
    # Load environment variables
    load_dotenv()
    
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config import Config
    from src.parser import ExcelParser
    
    parser = argparse.ArgumentParser(
        description='Generate BDD scenarios using Gemini',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate using User Story + Description (default)
  python -m src.generators.gemini_generator --prompt zero_shot --max 10
  
  # Generate using ONLY User Story (no description)
  python -m src.generators.gemini_generator --prompt zero_shot --user-story-only --max 10
  
  # Generate using ONLY Description (no user story)
  python -m src.generators.gemini_generator --prompt zero_shot --description-only --max 10
  
  # Generate with few-shot prompt (default)
  python -m src.generators.gemini_generator --prompt few_shot --max 10
  
  # Generate with chain-of-thought
  python -m src.generators.gemini_generator --prompt chain_of_thought
  
  # Custom input/output with description only
  python -m src.generators.gemini_generator --prompt few_shot --description-only --input data/Requirements500.xlsx
        """
    )
    
    parser.add_argument('--input', default='data/Requirements.xlsx')
    parser.add_argument('--output', help='Output file')
    parser.add_argument('--max', type=int, help='Max scenarios')
    parser.add_argument(
        '--prompt',
        default='few_shot',
        choices=['zero_shot', 'chain_of_thought', 'few_shot'],
        help='Prompt strategy to use (default: few_shot)'
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
        logger.info("BDD SCENARIO GENERATION - GEMINI")
        logger.info("=" * 80)
        
        config = Config.get_default_config()
        logger.info(f"Model: {config.gemini.model}")
        logger.info(f"Prompt strategy: {args.prompt}")
        
        logger.info("\nInitializing Gemini generator...")
        generator = GeminiGenerator(config.gemini, prompt_strategy=args.prompt)
        logger.info("✓ Generator initialized successfully")
        
        # Log the input mode being used
        if args.user_story_only:
            logger.info("\n⚠️  INPUT MODE: User Story ONLY (Description will be excluded)")
        elif args.description_only:
            logger.info("\n⚠️  INPUT MODE: Description ONLY (User Story will be excluded)")
        else:
            logger.info("\n✓ INPUT MODE: User Story + Description (default)")
        
        logger.info(f"\nReading requirements from: {args.input}")
        requirements = ExcelParser.read_requirements(Path(args.input))
        logger.info(f"✓ Loaded {len(requirements)} requirements")
        
        if args.max and args.max < len(requirements):
            requirements = requirements[:args.max]
            logger.info(f"Limiting to first {args.max} scenarios")
        
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
                results.append({
                    '#': requirement.id,
                    'User Story': requirement.user_story,
                    'Description': requirement.requirements,
                    'BDD-Reference': requirement.manual_scenario,
                    'BDD-AI Generated': scenario
                })
                logger.info("✓ Scenario generated successfully")
                logger.info(f"Preview: {scenario[:150]}...")
            except Exception as e:
                logger.error(f"✗ Failed: {e}")
                results.append({
                    '#': requirement.id,
                    'User Story': requirement.user_story,
                    'Description': requirement.requirements,
                    'BDD-Reference': requirement.manual_scenario,
                    'BDD-AI Generated': f"ERROR: {str(e)}"
                })
        
        results_df = pd.DataFrame(results)
        
        if args.output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if args.user_story_only:
                mode_suffix = "_user_story_only"
            elif args.description_only:
                mode_suffix = "_description_only"
            else:
                mode_suffix = ""
            args.output = f"results/gemini_scenarios_{args.prompt}{mode_suffix}_{timestamp}.xlsx"
        
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        results_df.to_excel(args.output, index=False)
        
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total: {len(requirements)}")
        logger.info(f"Success: {len([r for r in results if not r['BDD-AI Generated'].startswith('ERROR')])}")
        logger.info(f"Failed: {len([r for r in results if r['BDD-AI Generated'].startswith('ERROR')])}")
        logger.info(f"Output: {args.output}")
        logger.info("\n✓ Generation complete!")
        
    except KeyboardInterrupt:
        logger.info("\n\n✗ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
