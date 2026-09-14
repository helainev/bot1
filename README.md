# Local LLM Chatbot & LoRA Fine-Tuning Pipeline 💬🤖

A comprehensive GenAI repository focused on building an intelligent local chatbot integrated with the **Ollama API**, combined with an end-to-end parameter-efficient fine-tuning (PEFT) pipeline using **LoRA (Low-Rank Adaptation)**.

---

## 🛠️ Tech Stack & Tools

- **LLM Infrastructure:** Ollama API, Hugging Face Transformers
- **Fine-Tuning Techniques:** PEFT (LoRA), BitsAndBytes (Quantization)
- **Core Frameworks:** PyTorch, Python 3.12+
- **Environment Management:** `uv` (ultra-fast Python bundling) & Jupyter

---

## 🚀 Repository Structure

The project bridges the gap between deep learning model training and production-ready local inference:

- 🏋️‍♂️ **Model Fine-Tuning (`train_joker1.py` / `train_joker2.py`):** Python scripts and Jupyter environments leveraging LoRA to adapt a base open-source LLM on a custom dataset, significantly reducing VRAM requirements during training.
- ⚙️ **Data Preparation (`generate_custom.py`):** Automated utilities for synthetic data generation and prompt formatting tailored for effective model instruction-tuning.
- 🌐 **Inference App (`bot_app.py`):** A lightweight production script that establishes connection with the local Ollama instance to serve downstream chat interactions.

---

## ⚙️ Getting Started & Installation

Ensure you have a local instance of [Ollama](https://ollama.com) installed and running on your machine.

### Local Setup with `uv`

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd bot1
   ```

2. **Initialize environment and sync dependencies:**
   ```bash
   uv venv
   uv pip install -r pyproject.toml
   ```
   *Using `uv.lock` ensures an ultra-fast, reproducible installation of PyTorch, Transformers, and optimization libraries.*

---

## 🏋️‍♂️ Fine-Tuning Pipeline (LoRA)

The training pipeline applies **Low-Rank Adaptation** to inject trainable rank decomposition matrices into the LLM's attention layers. This allows domain adaptation without overwriting the base model's foundation weights.

To initiate the custom training sequence, prepare your text dataset and execute:
```bash
uv run python src/train_joker1.py
```

### Key Training Parameters Configured:
- **LoRA Rank ($r$):** Optimized parameter space constraints (typically 8 or 16).
- **Alpha ($\alpha$):** LoRA scaling factor for balancing base vs. adapter weights.
- **Quantization:** 4-bit/8-bit precision configurations via `bitsandbytes` to facilitate fine-tuning on consumer-grade hardware.

---

## 📡 Running the Chatbot

Once your custom weights are exported or integrated into an Ollama model definition file (`Modelfile`), trigger the interactive chat application:

```bash
uv run python bot_app.py
```
The script interacts directly with Ollama's local REST endpoints, enabling rapid prototyping, low-latency streaming responses, and complete data privacy.
