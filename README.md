# BDD Scenario Generator

An automated system for generating Behavior-Driven Development (BDD) scenarios using Large Language Models (LLMs) with comprehensive evaluation metrics.

## Overview

This tool generates BDD scenarios from requirements using multiple LLMs (GPT-4, Claude, Gemini) and evaluates them using both traditional text similarity metrics and LLM-based evaluation approaches.

## Features

- **Multiple LLM Support**: Generate scenarios using GPT-4, Claude 3.5, or Gemini
- **Flexible Prompting Strategies**: Zero-shot, Few-shot, and Chain-of-Thought prompting
- **Comprehensive Evaluation**: 
  - Text similarity metrics (BLEU, METEOR, ROUGE-L)
  - Semantic similarity (BERTScore, SBCS, SBED, USECS)
  - LLM-based evaluation (GPT-4, Claude, DeepSeek)
- **Modular Architecture**: Easy to extend with new generators or evaluators

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bdd_generator.git
cd bdd_generator
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLTK data (required for METEOR):
```bash
python setup_nltk.py
```

## Configuration

1. Copy the example configuration:
```bash
cp config.example.py config.py
```

2. Add your API keys to `config.py`:
```python
claude=ClaudeConfig(
    api_key="your-claude-api-key"
),
openai=OpenAIConfig(
    api_key="your-openai-api-key"
),
gemini=GeminiConfig(
    api_key="your-gemini-api-key"
)
```

## Usage

### Basic Usage

1. Prepare your input data in Excel/CSV format with columns:
   - `ID`: Unique identifier
   - `User Story`: User story
   - `Requirements`: Detailed description
   - `Manual Scenario`: Reference BDD scenario

2. Update the file path in `config.py`:
```python
files=FileConfig(
    requirements_path=Path("data/your_requirements.xlsx"),
    output_path=Path("results/comparison_results.csv")
)
```

3. Run the generator:
```bash
python main.py
```

### Using Different Generators

The repository includes generators for multiple LLMs. Example usage:

```python
from config import Config
from src.generators import ClaudeGenerator, GPT4Generator, GeminiGenerator
from src.parser import ExcelParser

# Load configuration
config = Config.get_default_config()

# Choose a generator
generator = ClaudeGenerator(config.claude)
# OR
generator = GPT4Generator(config.openai)
# OR
generator = GeminiGenerator(config.gemini)

# Read requirements
requirements = ExcelParser.read_requirements(config.files.requirements_path)

# Generate scenario
for req in requirements:
    scenario = generator.generate_scenario(req)
    print(scenario)
```

### Using Different Prompting Strategies

Prompts are externalized in the `prompts/` directory:
- `zero_shot.txt`: Direct generation without examples
- `few_shot.txt`: Generation with example scenarios
- `chain_of_thought.txt`: Step-by-step reasoning approach

To use a different strategy:

```python
from src.prompt_loader import PromptLoader

loader = PromptLoader()
prompt_template = loader.load_prompt("chain_of_thought")
```

### Evaluation

The system supports multiple evaluation approaches:

#### Text Similarity Metrics
```python
from src.comparators import BLEUComparator, METEORComparator, ROUGEComparator

comparator = BLEUComparator(config.bleu)
result = comparator.compare_scenarios(id, ai_scenario, manual_scenario)
```

#### Semantic Similarity
```python
from src.comparators import BERTComparator, SBCSComparator, USECSComparator

comparator = USECSComparator(config.usecs)
result = comparator.compare_scenarios(id, ai_scenario, manual_scenario)
```

#### LLM-based Evaluation
```python
from src.evaluators import GPT4Evaluator, ClaudeEvaluator, DeepSeekBDDEvaluator

evaluator = DeepSeekBDDEvaluator()
evaluation = evaluator.evaluate_scenario(ai_scenario, manual_scenario, requirements)
```

## Project Structure

```
bdd_generator/
├── src/
│   ├── generators/           # LLM-based scenario generators
│   │   ├── claude_generator.py
│   │   ├── gpt4_generator.py
│   │   └── gemini_generator.py
│   ├── evaluators/           # LLM-based evaluation
│   │   ├── claude_evaluator.py
│   │   ├── gpt4_evaluator.py
│   │   └── deepseek_evaluator.py
│   ├── comparators/          # Similarity metrics
│   │   ├── bleu_comparator.py
│   │   ├── meteor_comparator.py
│   │   ├── rouge_comparator.py
│   │   ├── bert_comparator.py
│   │   └── usecs_comparator.py
│   ├── models.py             # Data models
│   ├── parser.py             # Excel/CSV parsing
│   ├── prompt_loader.py      # Prompt management
│   └── csv_handler.py        # Results output
├── prompts/                  # Prompt templates
│   ├── zero_shot.txt
│   ├── few_shot.txt
│   └── chain_of_thought.txt
├── config.py                 # Configuration
├── main.py                   # Main entry point
└── requirements.txt          # Dependencies
```


