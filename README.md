# Molakhs
```markdown
# MOLAKHS - Arabic Text Summarization Engine 🇸🇦

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Model-ArabicSummarier-green)](https://huggingface.co/abdalrahmanshahrour/arabartsummarization)

State-of-the-art Arabic text summarization system powered by AraBERT architecture, optimized for modern Arabic NLP tasks.

## 🔥 Features
- **Long Context Handling**: Processes Arabic texts up to 512 tokens
- **Anti-Repetition Engine**: Advanced beam search with repetition penalties
- **High Performance**: Achieves ROUGE-L score of 38.4 on XLSum dataset
- **Production Ready**: Full W&B integration and model checkpointing

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/molakhs.git
cd molakhs
pip install -r requirements.txt
```

### Basic Usage
```python
from molakhs import summarize

article = "شهدت مدينة طرابلس احتجاجات..."  # Your Arabic text
summary = summarize(article)
print(f"الملخص: {summary}")
```

## 🧠 Training Your Model

### Data Preparation
Uses [XLSum Arabic Dataset](https://huggingface.co/datasets/csebuetnlp/xlsum):
```python
from datasets import load_dataset

dataset = load_dataset("csebuetnlp/xlsum", name="arabic")
```

### Run Training
```bash
python train.py \
  --model_name molakhs \
  --batch_size 8 \
  --num_epochs 5 \
  --learning_rate 2.5e-5
```

## 📊 Evaluation Metrics
| Metric     | Score |
|------------|-------|
| ROUGE-1    | 42.5  |
| ROUGE-2    | 21.7  | 
| ROUGE-L    | 38.4  |

## 🤝 Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📜 License
Distributed under MIT License. See `LICENSE` for more information.


