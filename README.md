# 🤖 NanoGPT: Transformer Language Model

A character-level Generative Pre-trained Transformer (GPT) built from scratch in PyTorch. 
This implementation constructs the Transformer architecture manually (Self-Attention, Multi-Head Attention, Feed-Forward Blocks) without using `nn.Transformer`.

## 🧠 Architecture Implementation
* **Embeddings:** Learned Token Embeddings + Positional Embeddings.
* **Self-Attention:** Implemented `Query`, `Key`, and `Value` matrix multiplications manually to calculate attention scores.
* **Masking:** Applied lower-triangular masking to ensure causal prediction (the model cannot see the future).
* **Residual Connections:** Implemented `Add & Norm` blocks for stable training.

## 📉 Results
Trained on Shakespeare text.
* **Initial Loss:** ~4.2 (Random guessing)
* **Final Loss:** ~1.8 (Coherent character patterns)

## 🚀 Usage
Run the script to train the model on the embedded text snippet and generate 200 characters of Shakespeare-like output.
