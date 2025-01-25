# MOLAKHS - Arabic Text Summarization Engine
# Production-Grade AI for Arabic NLP

%%capture
!pip install -q transformers datasets rouge-score wandb sentencepiece accelerate pyarabic

import torch
from pyarabic.tokenize import sentence_tokenize
from datasets import load_dataset, load_metric
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    set_seed
)
import wandb

# ===== MOLAKHS Configuration =====
CONFIG = {
    # Core Engine
    "model_name": "molakhs",
    "max_source_length": 512,
    "max_target_length": 128,
    
    # Arabic-Optimized Training
    "learning_rate": 2.5e-5,
    "batch_size": 8,
    "num_train_epochs": 5,
    
    # Advanced Generation
    "num_beams": 6,
    "repetition_penalty": 3.0,
    "length_penalty": 1.5,
    
    # Dataset
    "dataset_name": "csebuetnlp/xlsum",
    "dataset_config": "arabic",
    
    # Monitoring
    "wandb_project": "molakhs-ai"
}

# ===== Initialize MOLAKHS =====
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("abdalrahmanshahrour/arabartsummarization")
model = AutoModelForSeq2SeqLM.from_pretrained("abdalrahmanshahrour/arabartsummarization").to(device)

# ===== Data Processing =====
def load_molakhs_data():
    dataset = load_dataset(
        CONFIG["dataset_name"],
        name=CONFIG["dataset_config"],
        split="train"
    ).remove_columns(["id", "url", "title"])    
    return dataset.train_test_split(test_size=0.1, seed=42)

molakhs_data = load_molakhs_data().map(
    lambda examples: {
        "inputs": tokenizer(examples["text"], max_length=512, truncation=True, padding="max_length"),
        "labels": tokenizer(examples["summary"], max_length=128, truncation=True, padding="max_length")["input_ids"]
    },
    batched=True,
    remove_columns=["text", "summary"]
)

# ===== Training Engine =====
training_args = Seq2SeqTrainingArguments(
    output_dir="./molakhs",
    evaluation_strategy="epoch",
    learning_rate=CONFIG["learning_rate"],
    per_device_train_batch_size=CONFIG["batch_size"],
    num_train_epochs=CONFIG["num_train_epochs"],
    predict_with_generate=True,
    fp16=True,
    report_to="wandb",
    generation_max_length=CONFIG["max_target_length"],
    logging_steps=100,
    save_strategy="epoch"
)

# ===== Arabic Evaluation =====
def molakhs_metrics(eval_pred):
    preds, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Arabic-Specific Processing
    decoded_preds = [' '.join(sentence_tokenize(p.strip())) for p in decoded_preds]
    decoded_labels = [' '.join(sentence_tokenize(l.strip())) for l in decoded_labels]
    
    rouge = load_metric("rouge").compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    return {k: round(v.mid.fmeasure*100, 2) for k,v in rouge.items()}

# ===== Run MOLAKHS =====
wandb.init(project=CONFIG["wandb_project"])
Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=molakhs_data["train"],
    eval_dataset=molakhs_data["test"],
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=molakhs_metrics
).train().save_model("molakhs-ai")
wandb.finish()

# ===== MOLAKHS API =====
def molakhs_summarize(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
    return tokenizer.decode(
        model.generate(
            **inputs,
            num_beams=6,
            repetition_penalty=3.0,
            length_penalty=1.5,
            max_new_tokens=128
        )[0],
        skip_special_tokens=True
    )

# Example Usage
print(molakhs_summarize("[Your Arabic text here]"))
