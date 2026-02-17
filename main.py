import logging
from datetime import datetime
from typing import List
from config import Config
from src.parser import ExcelParser
from src.generators.claude_generator import ClaudeGenerator
from src.comparators.usecs_comparator import USECSComparator
from src.csv_handler import CSVHandler
from src.models import ComparisonResult

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BDDComparator:
    def __init__(self, config: Config):
        """Initialize the BDD comparison system"""
        self.config = config
        self.generator = ClaudeGenerator(config.claude)  # Use Claude generator
        self.comparator = USECSComparator(config.usecs)      # Use USECS comparator 

    def run(self) -> List[ComparisonResult]:
        """Run the complete BDD comparison process"""
        try:
            # Verify Excel structure
            logger.info("Verifying Excel structure...")
            ExcelParser.verify_excel_structure(self.config.files.requirements_path)
            
            # Read requirements
            logger.info("Reading requirements from Excel...")
            requirements = ExcelParser.read_requirements(self.config.files.requirements_path)
            
            # Validate requirements quality for Requirements field
            logger.info("Validating Requirements...")
            validation_warnings = ExcelParser.validate_requirements(requirements)
            if validation_warnings:
                logger.warning("Data quality warnings found:")
                for warning in validation_warnings[:10]:  # Show first 10 warnings
                    logger.warning(f"  - {warning}")
                if len(validation_warnings) > 10:
                    logger.warning(f"  ... and {len(validation_warnings) - 10} more warnings")
            else:
                logger.info("All Requirements passed validation checks")
            
            results = []
            total_usecs_score = 0
            
            # Process each requirement
            for req in requirements:
                logger.info(f"\nProcessing requirement {req.id}...")
                
                try:
                    # Add delay to avoid rate limits
                    import time
                    time.sleep(2)  # 2 second delay between requests
                    
                    # Generate AI scenario using Claude with Requirements only
                    logger.info("Generating scenario with Claude using Requirements only...")
                    logger.info(f"Requirements: {req.requirements[:100]}{'...' if len(req.requirements) > 100 else ''}")
                    ai_scenario = self.generator.generate_scenario(req)
                    logger.info("\nAI Generated Scenario:")
                    logger.info(f"{ai_scenario}")
                    
                    logger.info("\nManual Scenario:")
                    logger.info(f"{req.manual_scenario}")
                    
                    # Compare scenarios using USECS Score
                    comparison = self.comparator.compare_scenarios(
                        req.id,
                        ai_scenario,
                        req.manual_scenario
                    )
                    
                    usecs_score = comparison.overall_similarity
                    total_usecs_score += usecs_score
                    
                    logger.info(f"\nUSECS Score for Scenario {req.id}: {usecs_score:.2f}%")
                    
                    results.append(comparison)
                    
                except Exception as e:
                    logger.error(f"Error processing requirement {req.id}: {e}")
                    continue
            
            # Save results
            if results:
                avg_score = total_usecs_score / len(results)
                logger.info(f"\nAverage USECS Score: {avg_score:.2f}%")
                
                # Calculate statistics
                scores = [r.overall_similarity for r in results]
                min_score = min(scores) if scores else 0
                max_score = max(scores) if scores else 0
                
                logger.info("\nScore Statistics:")
                logger.info(f"Minimum Score: {min_score:.2f}%")
                logger.info(f"Maximum Score: {max_score:.2f}%")
                logger.info(f"Average Score: {avg_score:.2f}%")
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = self.config.files.output_path.parent / f"claude_usecs_results_{timestamp}.csv"
                CSVHandler.save_results(results, output_path)
                logger.info(f"\nResults saved to: {output_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in comparison process: {e}")
            raise

def main():
    """Main entry point"""
    try:
        logger.info(f"\n{'='*80}")
        logger.info("Starting BDD Scenario Generation and Comparison")
        logger.info("Using: Claude for Generation (with Requirements only)")
        logger.info("Using: USECS Score for Comparison")
        logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*80}")
        
        # Initialize configuration
        config = Config.get_default_config()
        
        # Create and run comparator
        comparator = BDDComparator(config)
        results = comparator.run()
        
        if results:
            logger.info(f"\nProcessed {len(results)} scenarios")
            logger.info(f"Execution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            logger.warning("No results generated!")
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        raise

if __name__ == "__main__":
    main()