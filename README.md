# Quantized-Mistral-Model-Loading

Minimal example for loading a quantized Mistral GGUF model with `llama-cpp-python` and generating a response from the command line.

## File

- `simple_mistral_example.py` — loads `Mistral-7B-Instruct-v0.3.Q4_K_M.gguf` and runs an interactive prompt.

## Requirements

- Python 3
- `llama-cpp-python`
- A local GGUF model file at:

	`/home/user/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf`

Install the Python package:

```bash
pip install llama-cpp-python
```

## Run

```bash
python3 simple_mistral_example.py
```

Then enter a prompt when asked.

## What the script does

- verifies that `llama-cpp-python` is installed
- checks that the GGUF model file exists
- loads the model with an 8k context window
- accepts a user question from standard input
- generates and prints the model response
- shows token usage statistics

## Notes

- The script is currently configured for CPU inference with `n_gpu_layers=0`.
- Update the model path in the script if your GGUF file is stored elsewhere.
