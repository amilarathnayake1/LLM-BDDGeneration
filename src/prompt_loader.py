"""
Prompt Loader Utility

Loads BDD generation prompts from external files.
All LLMs share the same common prompts.
"""

from pathlib import Path
from typing import List

class PromptLoader:
    """Load and manage BDD generation prompts"""
    
    def __init__(self, prompts_dir: str = "prompts"):
        """Initialize prompt loader with base directory"""
        self.prompts_dir = Path(prompts_dir)
    
    def load_prompt(self, strategy: str = "zero_shot") -> str:
        """
        Load a prompt template from file
        
        Args:
            strategy: Prompting strategy (zero_shot, chain_of_thought, few_shot)
            
        Returns:
            Prompt template string with {user_story} and {description} placeholders
            
        Raises:
            FileNotFoundError: If prompt file doesn't exist
        """
        prompt_file = self.prompts_dir / f"{strategy}.txt"
        
        if not prompt_file.exists():
            available = self.list_available_prompts()
            raise FileNotFoundError(
                f"Prompt file not found: {prompt_file}\n"
                f"Available prompts: {available}"
            )
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    def list_available_prompts(self) -> List[str]:
        """List all available prompt files"""
        if not self.prompts_dir.exists():
            return []
        
        return [f.stem for f in self.prompts_dir.glob("*.txt")]
    
    def create_prompt(self, user_story: str, description: str, template: str) -> str:
        """
        Create a prompt by filling in the template with actual data
        
        Args:
            user_story: The user story text
            description: The requirements description
            template: The prompt template with {user_story} and {description} placeholders
            
        Returns:
            Filled prompt ready to send to LLM
        """
        return template.format(
            user_story=user_story,
            description=description
        )


# Convenience function for quick access
def load_prompt(strategy: str = "zero_shot") -> str:
    """Quick function to load a prompt"""
    loader = PromptLoader()
    return loader.load_prompt(strategy)
