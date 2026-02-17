import pandas as pd
from pathlib import Path
from typing import List
from .models import RequirementData

class ExcelParser:
    @staticmethod
    def read_requirements(file_path: Path) -> List[RequirementData]:
        """Read requirements from Excel or CSV file - supporting Requirements for GPT4 generator"""
        try:
            # Check file extension and read accordingly
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            else:
                raise Exception(f"Unsupported file format: {file_path.suffix}")
                
            requirements_list = []
            
            for _, row in df.iterrows():
                # Read Requirements for GPT4 generator, User Story optional
                req = RequirementData(
                    id=str(row['ID']),
                    user_story=str(row.get('User Story', '')),  # Optional - set to empty if not present
                    requirements=str(row.get('Requirements', '')),  # Used by GPT-4 generator
                    manual_scenario=str(row['Manual Scenario'])
                )
                requirements_list.append(req)
                
            return requirements_list
            
        except Exception as e:
            raise Exception(f"Error reading file: {e}")

    @staticmethod
    def verify_excel_structure(file_path: Path) -> bool:
        """Verify file has required columns for GPT4 Requirements-based BDD generation"""
        try:
            # Check file extension and read accordingly
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            else:
                raise Exception(f"Unsupported file format: {file_path.suffix}")
                
            # Essential columns for GPT4 Requirements-based BDD generation
            essential_columns = ['ID', 'Requirements', 'Manual Scenario']
            
            missing_essential = [col for col in essential_columns if col not in df.columns]
            if missing_essential:
                raise Exception(f"Missing essential columns: {missing_essential}")
            
            # Check for empty Requirements since they are critical for GPT-4 generator
            if df['Requirements'].isnull().any() or (df['Requirements'] == '').any():
                raise Exception("Found empty Requirements entries. Requirements are required for GPT-4 generator.")
                
            return True
            
        except Exception as e:
            raise Exception(f"Error verifying file structure: {e}")
    
    @staticmethod
    def validate_requirements(requirements: List[RequirementData]) -> List[str]:
        """Validate Requirements for GPT4 BDD generation quality"""
        warnings = []
        
        for req in requirements:
            requirements_text = req.requirements.strip()
            
            # Validate Requirements (for GPT-4 generator)
            if not requirements_text or requirements_text.lower() == 'nan':
                warnings.append(f"ID {req.id}: Missing Requirements - required for GPT-4 generator")
            elif len(requirements_text) < 20:
                warnings.append(f"ID {req.id}: Requirements might be too short for comprehensive BDD generation")
            
            # Check for clear functional requirements
            requirements_lower = requirements_text.lower()
            if not any(phrase in requirements_lower for phrase in ['should', 'must', 'will', 'needs to', 'required to', 'expected to']):
                warnings.append(f"ID {req.id}: Requirements lack clear requirement indicators (should, must, will, etc.)")
            
            # Check for action words that help with BDD scenario generation
            if not any(action in requirements_lower for action in ['create', 'view', 'edit', 'delete', 'update', 'add', 'remove', 'login', 'register', 'search', 'filter', 'sort', 'submit', 'save', 'cancel', 'upload', 'download', 'process', 'validate', 'verify', 'access', 'navigate', 'select', 'click', 'enter', 'manage']):
                warnings.append(f"ID {req.id}: Requirements lack clear action verbs for BDD scenario generation")
            
            # Check for system behavior indicators
            if not any(behavior in requirements_lower for behavior in ['system', 'application', 'user', 'display', 'show', 'present', 'provide', 'enable', 'allow', 'prevent', 'ensure', 'interface', 'page', 'screen', 'form', 'button', 'field']):
                warnings.append(f"ID {req.id}: Requirements lack clear system behavior indicators")
        
        return warnings