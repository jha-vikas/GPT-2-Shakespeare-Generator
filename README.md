# Shakespeare GPT-2 (124M) - Overfitted to Loss < 0.1

This repository contains a from-scratch implementation and training of a **GPT-2 (124M)** decoder-only transformer. The model was trained specifically to "memorize" a 1.1MB Shakespeare dataset (`input.txt`), achieving a final loss of **< 0.1** as required by the assignment.

## 🚀 Live Demo
You can interact with the trained model here:
**[Hugging Face Space: Shakespeare GPT-2](https://huggingface.co/spaces/boy1729/gpt2_custom_trained)**

---

## 📂 Repository Structure
*   `train_get2-8-init_updated.py`: Optimized training script with Cosine Decay LR and Gradient Clipping.
*   `training_log.txt`: Complete logs showing the training process and the target loss being achieved.
*   `app.py`: The Gradio application script used for the Hugging Face deployment.
*   `requirements.txt`: Python dependencies for both training and deployment.
*   `model.pt`: (Optional/Reference) The saved master weights of the model (~500MB).

---

## 🛠️ Model & Training Details

### Architecture
- **Parameters**: 124.35 Million
- **Layers**: 12
- **Heads**: 12
- **Embedding Dim**: 768
- **Context Size**: 1024 (trained on 128 for speed)
- **Vocabulary**: 50,257 (GPT-2 Tiktoken)

### Technical Optimizations
To achieve the extremely low loss of **0.0997** on a Mac M3 Max, the following techniques were used:
- **Optimizer**: AdamW ($\beta_1=0.9, \beta_2=0.95$)
- **Learning Rate**: $3 \times 10^{-4}$ with **Cosine Decay** down to $3 \times 10^{-5}$.
- **Gradient Clipping**: Global norm clipped at `1.0` to ensure stability during the final memorization phase.
- **Precision**: `bfloat16` Mixed Precision for high-speed training on Apple Silicon (MPS).
- **Weight Sharing**: Token embeddings and LM Head weights are tied to reduce parameter count and improve learning.

---

## 📈 Training Progress
The model was trained for approximately **4,200 steps** on a **Mac M3 Max**. 

**Final Steps Log:**
```text
step  4160 | loss: 0.101243 | lr: 2.84e-04 | norm: 1.0543 | dt: 295.40ms
step  4170 | loss: 0.100582 | lr: 2.84e-04 | norm: 1.0210 | dt: 298.15ms
step  4180 | loss: 0.100120 | lr: 2.84e-04 | norm: 0.9854 | dt: 296.80ms

Target loss 0.099703 reached at step 4187!
Model saved to model.pt
```
*The full training logs can be found in [training_log.txt](./training_log.txt).*

---

## 🎭 Sample Output
After reaching the target loss, the model was prompted with `"ROMEO:"`. Here is a sample of the output:

```text
ROMEO:
No truly sister, to each other's heavy stay,
To question, and make a sudden sorrow
To a most wonderful house, and a most sweet
And a most happy time.
```

---

## 📸 Hugging Face Interface
![Hugging Face Space Screenshot](./screenshot.png)
*(Note: Replace this placeholder with your actual screenshot from the Space)*

---

## 📖 How to Run
1.  **Clone the repo**:
    ```bash
    git clone https://github.com/[your-username]/[your-repo-name]
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run training**:
    ```bash
    python -u train_get2-8-init_updated.py | tee training_log.txt
    ```

---

### Links Referenced
*   [Hugging Face Space](https://huggingface.co/spaces/boy1729/gpt2_custom_trained)
*   [Tiktoken Encoding](https://github.com/openai/tiktoken)
*   [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)

