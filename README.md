# PInG

## Overview

PInG is a code generation framework that leverages code comments as a medium for developers and LLMs to establish a shared understanding. This project is the code repository for PInG. It includes modules for data preprocessing, model fine-tuning, evaluation, and generation, among other tasks.

## Key Features

- **Comment Analysis**: Tools to preprocess, train, and evaluate models on comment data.
- **Evaluation Suite**: Comprehensive scripts for assessing model performance on HumanEval, MBPP, and other benchmarks.
- **Fine-Tuning Framework**: Scripts and configurations for fine-tuning models on custom datasets.
- **Generation Utilities**: Scripts for text/code generation using various models.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/repository.git
   cd repository
   ```

2. Install dependencies:
   ```bash
   pip install -r finetune/dataset/preprocessing/requirements.txt
   ```

## Usage

### Fine-Tuning
```bash
python finetune/finetune.py --config finetune/configs/ds_config_zero3.json
```

### Evaluation
```bash
python evaluate/evaluate.py --dataset evaluate/data/humaneval.py
```

### Generation
```bash
python generate/generate.py --model generate/model.py
```

## Contributing

Contributions are welcome! Open an issue or submit a pull request following our contribution guidelines.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
