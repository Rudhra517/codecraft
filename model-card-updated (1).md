# Code Craft Model Card

## Model Details

- **Model Name:** Code Craft
- **Model Type:** Distilled Code LLM (Large Language Model)
- **Version:** 1.0
- **Developers:** The Code Craft Team (Led by Rudra)
- **Release Date:** April 2025
- **Repository:** [https://huggingface.co/rudra157/codecraft](https://huggingface.co/rudra157/codecraft)
- **License:** MIT

## Model Architecture

Code Craft is a distilled language model specialized for code generation and completion tasks. The model was created using knowledge distillation techniques to reduce the size of larger foundation models while maintaining coding capabilities.

- **Base Architecture:** Transformer-based decoder-only
- **Parameter Count:** ~350M parameters (95% reduction from teacher model)
- **Context Length:** 4096 tokens
- **Training Method:** Knowledge distillation + LoRA fine-tuning
- **Teacher Model:** CodeLlama-7B
- **Available Formats:** GGUF (various quantization levels)

## Intended Use

Code Craft is designed for:

- Code completion and generation
- Function implementation based on descriptions or signatures
- Code documentation generation
- Simple debugging and error correction
- Coding assistance for educational purposes

The model works best for Python code but has capabilities in other programming languages including JavaScript, Java, C++, and Go.

## Training Methodology

### Distillation Process

The model was trained using a multi-stage approach:

1. **Architecture Reduction:** Modified the teacher model architecture to reduce attention heads and hidden layer dimensions
2. **Knowledge Distillation:** The student model (Code Craft) was trained to mimic the output distributions of the teacher model
3. **Fine-tuning:** Additional fine-tuning on code-specific datasets to regain performance lost during distillation

### Training Data

The model was trained on a combination of:

- Code completion pairs extracted from public repositories
- Programming problem statements and solutions
- Documentation and implementation pairs
- Code snippets with accompanying explanations

### Optimization

Training utilized several optimization techniques:

- **Unsloth Library:** Used for efficient training optimization
- **LoRA (Low-Rank Adaptation):** Parameter-efficient fine-tuning
- **Quantization-Aware Training:** Prepared the model for efficient deployment

## Performance and Limitations

### Benchmarks

| Benchmark | Score | Comparison to Teacher |
|-----------|-------|------------------------|
| HumanEval | 42.3% | 93.5% of teacher model |
| MBPP      | 38.7% | 91.2% of teacher model |
| CodeXGLUE | 47.6% | 94.8% of teacher model |

### Quantitative Analysis

- **Speed:** 5-10x faster inference than the teacher model
- **Memory:** Requires ~1GB RAM in q4_k_m quantization
- **Performance per Parameter:** Approximately 4.2x more efficient

### Limitations

- Less effective for very complex algorithms or larger codebases
- Reduced understanding of obscure programming languages
- May occasionally produce incorrect code for edge cases
- Limited context window compared to some larger models
- Does not have knowledge of very recent programming languages, frameworks, or libraries released after training

## Ethical Considerations

- **Educational Use:** The model is primarily intended for educational purposes and coding assistance
- **Code Quality:** Generated code should be reviewed for correctness and security
- **Attribution:** The model may occasionally reproduce code patterns from its training data
- **Bias:** May reflect biases present in the training data, such as preferences for certain coding styles or solutions

## Quantization Variants

The model is available in multiple quantization formats:

- **codecraft-q4_k_m.gguf:** Balanced quality and size (default)
- **codecraft-q5_k_m.gguf:** Higher quality, larger size
- **codecraft-q2_k.gguf:** Smallest file size, reduced quality

## Usage Examples

```python
from llama_cpp import Llama

# Initialize the model
model = Llama(
    model_path="path/to/codecraft-q4_k_m.gguf",
    n_ctx=4096,
    n_threads=4
)

# Example code completion
prompt = "def calculate_fibonacci(n):"
response = model(prompt, max_tokens=200, temperature=0.7)
print(response['choices'][0]['text'])
```

## Citation

If you use this model in your research, please cite:
```
@software{codecraft_2025,
  author = {Rudra and The Code Craft Team},
  title = {Code Craft: Efficient Model Distillation for Code Generation},
  year = {2025},
  url = {https://huggingface.co/rudra157/codecraft}
}
```

## Team

This model was developed by the Code Craft Team:
- Rudra (Lead Developer)
- [Other team member names]

The project was created as part of the Advanced AI course at [Your University].
