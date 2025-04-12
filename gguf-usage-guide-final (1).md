# Using Code Craft GGUF Models

This guide explains how to download and use the Code Craft model in GGUF format from Hugging Face.

## About Code Craft

Code Craft is a distilled code generation model developed by the Code Craft team, led by Rudra. The model is designed to be efficient and lightweight while maintaining excellent code generation capabilities.

## What is GGUF?

GGUF (GPT-Generated Unified Format) is an efficient file format for machine learning models, optimized for local inference. It offers improved memory usage, faster loading times, and better quantization compared to older formats like GGML.

## Installation Requirements

Before using the Code Craft GGUF model, install the following dependencies:

```bash
# Install llama.cpp Python bindings
pip install llama-cpp-python

# For GPU acceleration (CUDA)
pip install llama-cpp-python-cuda

# For Apple Silicon
pip install llama-cpp-python-metal
```

## Downloading the Model

### Option 1: Using Hugging Face CLI

```bash
# Install Hugging Face Hub CLI
pip install huggingface_hub

# Download the model
huggingface-cli download rudra157/codecraft codecraft-q4_k_m.gguf
```

### Option 2: Python API

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="rudra157/codecraft",
    filename="codecraft-q4_k_m.gguf"
)
print(f"Model downloaded to: {model_path}")
```

### Option 3: Manual Download

1. Visit `https://huggingface.co/rudra157/codecraft`
2. Navigate to the Files tab
3. Download `codecraft-q4_k_m.gguf` to your local machine

## Using the Model

### Basic Usage with llama.cpp Python

```python
from llama_cpp import Llama

# Initialize the model
model = Llama(
    model_path="/path/to/codecraft-q4_k_m.gguf",
    n_ctx=4096,  # Context window size
    n_threads=4  # Number of CPU threads to use
)

# Generate code completion
prompt = "def fibonacci(n):"
response = model(
    prompt,
    max_tokens=200,
    temperature=0.7,
    top_p=0.95,
    echo=True
)

print(response['choices'][0]['text'])
```

### Advanced Usage - Interactive Code Completion

```python
import textwrap
from llama_cpp import Llama

# Load the model
model = Llama(
    model_path="/path/to/codecraft-q4_k_m.gguf",
    n_ctx=4096,
    n_threads=4
)

def complete_code(prompt, max_new_tokens=200):
    """Generate code completion with the Code Craft model."""
    response = model(
        prompt,
        max_tokens=max_new_tokens,
        temperature=0.5,
        top_p=0.9,
        stop=["```"],
        echo=False
    )
    return response['choices'][0]['text']

# Example usage
if __name__ == "__main__":
    code_prompt = """
    # Write a function to find the longest common substring
    # between two strings
    def longest_common_substring(str1, str2):
    """
    
    result = complete_code(textwrap.dedent(code_prompt))
    print(f"Prompt:\n{code_prompt}\n")
    print(f"Completion:\n{result}")
```

## Using with Text-Generation WebUI

The Code Craft GGUF model is compatible with [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui), a popular interface for running LLMs locally.

1. Install text-generation-webui following their instructions
2. Place your GGUF file in the `models` directory
3. Start the web UI: `python server.py`
4. Select "Code Craft" from the model dropdown
5. Configure the parameters as needed for code generation

## Performance Tuning