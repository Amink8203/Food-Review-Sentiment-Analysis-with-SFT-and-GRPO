# Persian Food Review Sentiment Analysis with LLM Fine-tuning

## Project Overview

This project focuses on sentiment analysis of Persian food delivery customer reviews using advanced Language Model (LLM) fine-tuning techniques. The project compares three different approaches: a baseline model, Supervised Fine-Tuning (SFT), and Group Relative Policy Optimization (GRPO) to classify customer sentiments as either "happy" or "sad".

## 📊 Dataset

The project uses a Persian food delivery customer review dataset with the following characteristics:

- **Training Set:** 56,700 samples
- **Development Set:** 6,300 samples  
- **Test Set:** 7,000 samples
- **Languages:** Persian (Farsi)
- **Classes:** Binary classification (HAPPY/SAD)
- **Domain:** Food delivery customer reviews

### Data Distribution
The dataset contains customer reviews from food delivery services with balanced sentiment distribution, capturing various aspects of customer experience including food quality, delivery time, service quality, and overall satisfaction.

## 🏗️ Model Architecture

**Base Model:** Google Gemma-3-1B-IT
- **Architecture:** Instruction-tuned variant of Gemma-3
- **Parameters:** 1 billion parameters
- **Context Length:** 2048-4000 tokens depending on the task
- **Quantization:** 4-bit quantization for efficient training

## 🚀 Methodology

### 1. Supervised Fine-Tuning (SFT)
- **Technique:** LoRA (Low-Rank Adaptation) fine-tuning
- **Target Modules:** Query, Key, Value, Output projections + MLP layers
- **LoRA Rank:** 16
- **Training Strategy:** Response-only training for efficiency
- **Optimizer:** AdamW 8-bit
- **Learning Rate:** 2e-4

### 2. Group Relative Policy Optimization (GRPO)
- **Approach:** Reinforcement Learning from Human Feedback (RLHF)
- **Reward Functions:**
  - **Correct Answer Reward:** +20 for correct sentiment, -20 for incorrect
  - **Format Reward:** Proper XML tag usage validation
  - **Word Count Reward:** Reasoning length constraint (≤55 words)
- **Training Configuration:**
  - Learning Rate: 1e-5
  - Batch Size: 8 with gradient accumulation
  - Generations per prompt: 2

### 3. Chat Template System
Custom system prompts designed for Persian sentiment analysis:
```
You are an expert sentiment‑analysis assistant for Persian food‑delivery customer reviews.
- Read the user's comment
- Decide which single overall emotion is conveyed
Valid answers: happy (positive/satisfied) or sad (negative/dissatisfied)
```

## 📈 Results

### Sentiment Analysis Performance

| Model | Accuracy | F1-Score (Macro) | Precision | Recall |
|-------|----------|------------------|-----------|---------|
| **Base Model** | ~45% | ~0.42 | - | - |
| **SFT Model** | ~85% | ~0.83 | 0.82 | 0.84 |
| **GRPO Model** | **80.0%** | **0.79** | 0.81 | 0.79 |

### Key Findings

1. **SFT Superior Performance:** SFT achieved the best results for this sentiment classification task
2. **GRPO Reasoning Benefits:** While GRPO showed slightly lower sentiment accuracy, it demonstrated better reasoning capabilities
3. **Significant Improvement:** Both fine-tuned models substantially outperformed the base model
4. **Domain Adaptation:** Fine-tuning successfully adapted the model to Persian food review vocabulary and patterns

### Confusion Matrix Analysis
- **SFT Model:** Better balance between precision and recall
- **GRPO Model:** Slight bias toward SAD class detection (89% recall for SAD vs 69% for HAPPY)

## 🧮 Mathematical Reasoning Evaluation (AIME 2024)

To test generalization and reasoning capabilities, models were evaluated on the AIME 2024 mathematical problem dataset:

| Model | Problems Solved | Accuracy |
|-------|----------------|----------|
| Base Model | 0/30 | 0% |
| SFT Model | 0/30 | 0% |
| GRPO Model | 1/30 | 3.3% |

**Key Insight:** GRPO's reward-based training for reasoning showed marginal improvement in mathematical problem-solving, demonstrating the potential of reinforcement learning for complex reasoning tasks.

## 🛠️ Technical Implementation

### Dependencies
```python
- unsloth (LLM fine-tuning framework)
- transformers (Hugging Face)
- trl (Transformer Reinforcement Learning)
- torch (PyTorch)
- peft (Parameter Efficient Fine-Tuning)
- datasets (Data handling)
- sklearn (Evaluation metrics)
```

### Training Configuration
- **GPU:** Compatible with T4/V100/A100 GPUs
- **Memory Optimization:** 4-bit quantization + gradient checkpointing
- **Batch Processing:** Dynamic batching with padding
- **Evaluation:** Step-based evaluation every 10 steps

### Data Processing Pipeline
1. **Text Preprocessing:** Persian text normalization and cleaning
2. **Chat Formatting:** Structured conversation format with system/user/assistant roles
3. **Tokenization:** Gemma-3 specific tokenization with special tokens
4. **Response Masking:** Training only on model responses for efficiency

## 📊 Evaluation Metrics

- **Accuracy:** Overall classification correctness
- **F1-Score (Macro):** Balanced measure considering both classes
- **Precision/Recall:** Per-class performance analysis
- **Confusion Matrix:** Detailed error analysis
- **Classification Report:** Comprehensive per-class metrics

## 🔍 Error Analysis

### Common Misclassifications
1. **Mixed Sentiment:** Reviews with both positive and negative aspects
2. **Sarcasm/Irony:** Subtle Persian linguistic nuances
3. **Context-Dependent:** References requiring domain knowledge
4. **Ambiguous Expressions:** Neutral or unclear sentiment indicators

## 📋 Usage Instructions

### Setup
```bash
# Install dependencies
pip install unsloth vllm transformers trl torch peft datasets scikit-learn

# For Colab users
pip install --no-deps unsloth vllm==0.8.5.post1
```

### Training
```python
# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-3-1b-it",
    max_seq_length=2048,
    load_in_4bit=False
)

# Configure LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# Train with SFT
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset
)
```

### Inference
```python
# Generate sentiment prediction
prompt = build_prompt("خیلی خوب بود")
prediction = evaluate(model, tokenizer, prompt)
```

## 📁 Project Structure

```
Snappfood_llm_proj/
├── project.ipynb                            # Main notebook
├── train.csv                                # Training dataset
├── dev.csv                                  # Development dataset  
├── test.csv                                 # Test dataset
└── README.md                                # This file
```

## 🎯 Conclusion

This project demonstrates the effectiveness of fine-tuning approaches for Persian sentiment analysis:

1. **SFT proved most effective** for this specific sentiment classification task
2. **GRPO showed promise** for reasoning-intensive tasks despite lower sentiment accuracy
3. **Domain-specific fine-tuning** significantly outperformed general-purpose models
4. **Persian NLP capabilities** were successfully enhanced through targeted training

The project provides valuable insights into adapting large language models for low-resource languages and domain-specific applications, with practical implications for Persian e-commerce and customer service applications.
