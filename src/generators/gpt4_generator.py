from openai import OpenAI
from typing import List
from ..models import RequirementData
from config import OpenAIConfig

class GPT4Generator:
    def __init__(self, config: OpenAIConfig):
        """Initialize the BDD generator with OpenAI configuration"""
        self.config = config
        self.client = OpenAI(api_key=config.api_key)

    def generate_scenario(self, requirement: RequirementData) -> str:
        """Generate BDD scenario using OpenAI API based on User Stories and Requirements"""
        try:
            prompt = self._create_prompt(requirement)
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are a QA expert who writes comprehensive BDD scenarios based on detailed technical requirements and specifications."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise Exception(f"Error generating BDD scenario: {e}")

    def _create_prompt(self, requirement: RequirementData) -> str:
        """Create the prompt for scenario generation"""
        return f"""You are a QA expert. Generate a BDD scenario for the given user story and description using the examples below as guidance.

Example 1:
User Story: As a marketing manager, I want to upload brand assets to the system so that my team can access approved materials.
Description: The system should allow users with appropriate permissions to upload image files (PNG, JPG, JPEG) up to 10MB in size. The uploaded files should be automatically categorized and made available to team members.
BDD Scenario:
Scenario: Marketing manager uploads brand logo successfully
Given I am logged in as a marketing manager with upload permissions
And I am on the asset upload page
When I select a PNG file that is 5MB in size
And I click the "Upload" button
Then the file should be uploaded successfully
And I should see a confirmation message "File uploaded successfully"
And the file should appear in my brand assets library

Example 2:
User Story: As a content creator, I want to search for existing assets by keyword so that I can quickly find relevant materials for my projects.
Description: Users should be able to enter keywords in a search box and receive filtered results showing matching assets. The search should work across file names, tags, and descriptions with results displayed in a grid format.
BDD Scenario:
Scenario: Content creator searches for logo assets
Given I am logged in as a content creator
And there are assets tagged with "logo" in the system
When I enter "logo" in the search box
And I click the search button
Then I should see a list of assets matching "logo"
And the results should be displayed in a grid format
And each result should show the asset thumbnail and name

Example 3:
User Story: As a system admin, I want to assign user permissions to team members so that I can control who can access specific brand assets.
Description: Admin should be able to select users from a list and assign them view, edit, or admin permissions for specific asset folders. Changes should take effect immediately.
BDD Scenario:
Scenario: Admin assigns view permissions to team member
Given I am logged in as a system administrator
And I have a team member "John Smith" in the user list
And there is a folder called "Brand Guidelines"
When I select "John Smith" from the user list
And I assign "view" permissions for the "Brand Guidelines" folder
And I click "Save Changes"
Then John Smith should have view access to the Brand Guidelines folder
And I should see a confirmation message "Permissions updated successfully"

Your Task:
Now generate one BDD scenario for this user story and description:
User Story: {requirement.user_story}
Description: {requirement.requirements}

Use the following format:
Scenario: [Scenario name]
Given [precondition]
When [action]
Then [expected result]

DO NOT mention any special notes or alternative scenarios. Just one scenario only."""


if __name__ == "__main__":
    """Run GPT-4 generator from command line"""
    import sys
    import argparse
    from pathlib import Path
    from datetime import datetime
    import pandas as pd
    from dotenv import load_dotenv
    import os
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load environment variables
    load_dotenv()
    
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config import Config
    from src.parser import ExcelParser
    
    parser = argparse.ArgumentParser(
        description='Generate BDD scenarios using GPT-4',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate for all requirements
  python -m src.generators.gpt4_generator
  
  # Generate for first 10 requirements only
  python -m src.generators.gpt4_generator --max 10
        """
    )
    
    parser.add_argument('--input', default='data/Requirements.xlsx')
    parser.add_argument('--output', help='Output file')
    parser.add_argument('--max', type=int, help='Max scenarios')
    
    args = parser.parse_args()
    
    try:
        logger.info("=" * 80)
        logger.info("BDD SCENARIO GENERATION - GPT-4")
        logger.info("=" * 80)
        
        config = Config.get_default_config()
        logger.info(f"Model: {config.openai.model}")
        
        logger.info("\nInitializing GPT-4 generator...")
        generator = GPT4Generator(config.openai)
        logger.info("✓ Generator initialized successfully")
        
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
            args.output = f"results/gpt4_scenarios_{timestamp}.xlsx"
        
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