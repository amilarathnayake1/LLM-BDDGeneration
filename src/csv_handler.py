import pandas as pd
from pathlib import Path
from typing import List
from datetime import datetime
from .models import ComparisonResult

class CSVHandler:
    @staticmethod
    def save_results(results: List[ComparisonResult], output_path: Path) -> None:
        """Save comparison results to CSV file with timestamp in filename"""
        try:
            # Get current timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create filename with timestamp
            output_filename = f"comparison_results_{timestamp}.csv"
            final_path = output_path.parent / output_filename
            
            # Convert results to dictionaries
            data = [result.to_dict() for result in results]
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame(data)
            df.to_csv(final_path, index=False)
            
            return final_path
            
        except Exception as e:
            raise Exception(f"Error saving results to CSV: {e}")