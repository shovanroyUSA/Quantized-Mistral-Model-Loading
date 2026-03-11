#!/usr/bin/env python3
"""
Simple Mistral-7B Model Example
================================

This program loads a pre-trained Mistral model and generates a response
to a simple question using the installed GGUF model file.

Requirements:
  - llama-cpp-python library (install: pip install llama-cpp-python)
  - Mistral-7B-Instruct-v0.3.Q4_K_M.gguf model file in /home/user/models/

Output:
  Prints the model's response to "What is the capital of India?"
"""

from pathlib import Path

# Step 1: Try to import the library that can load GGUF model files
try:
    from llama_cpp import Llama
except ImportError:
    print("ERROR: llama-cpp-python is not installed!")
    print("Install it with: pip install llama-cpp-python")
    exit(1)

# Step 2: Define the path to the model file
# The model file should be in /home/user/models/ directory
model_path = Path("/home/user/models") / "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"
#model_path = Path("/home/user/models") / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# Step 3: Check if the model file exists
if not model_path.exists():
    print(f"ERROR: Model file not found at {model_path}")
    print(f"Make sure the model file is downloaded to {model_path}")
    exit(1)

print("=" * 70)
print("Loading Mistral-7B model...")
print("=" * 70)

# Step 4: Load the model from the GGUF file
# This creates a 'mistral_model' object that we can use to generate text
# n_gpu_layers: Number of layers to offload to GPU (if available)
#               -1 means all layers (use GPU if available)
# n_ctx: Context window size (how much text the model can "remember")
# n_threads: Number of CPU threads to use
mistral_model = Llama(
    model_path=str(model_path),
    n_gpu_layers=0,# Use GPU if available
    n_ctx=8192,   # Context size: 8k tokens upto 32k supported
    n_threads=8,          # Use 8 CPU threads
    verbose=False         # Don't print debug info
)

print(f"✓ Model loaded successfully!\n")

# Step 5: Create a prompt/question to ask the model
question = input("Enter a question for the Mistral model (or press Enter for default): ").strip()

print(f"Question: {question}\n")
print("=" * 70)
print("Generating response...")
print("=" * 70)

# Step 6: Generate a response from the model
# max_tokens: Maximum number of tokens (words/pieces) to generate
# temperature: How "creative" the output should be (0.0 = deterministic, 1.0 = creative)
response = mistral_model(
    prompt=question,
    max_tokens = 2048,# Generate up to 2048 tokens
    temperature=0.7,     # Balanced creativity
    top_p=0.95,          # Nucleus sampling parameter
   #stop=["\n\n"], # Stop tokens to prevent the model from rambling
    echo=False           # Don't repeat the input prompt in output
)

# Step 7: Extract the generated text from the response
# The response is a dictionary, and the text is in the 'choices' list
generated_text = response["choices"][0]["text"].strip()

# Step 8: Print the response
print(f"\nResponse:")
print("-" * 70)
print(generated_text)
print("-" * 70)

# Step 9: Print some metadata about the generation
print(f"\nTokens generated: {response['usage']['completion_tokens']}")
print(f"Total tokens used: {response['usage']['total_tokens']}")
print("\n✓ Generation complete!")
