# Sequence Classification with BERT and Custom GRU with Attention

This project implements two different approaches for sequence classification using natural language processing (NLP) techniques: a BERT-based model and a custom-built GRU model with attention mechanism. The goal of this project is to classify text sequences (e.g., user comments) into two distinct categories.

## Project Overview
1. **BERT-based Model:** Utilizes the pre-trained bert-base-cased model from Hugging Face for sequence classification, fine-tuned on a dataset of labeled comments.
2. **Custom GRU Model with Attention:** A self-designed model incorporating a GRU (Gated Recurrent Unit) with an attention layer for capturing important words and classifying text sequences.

Both models are trained, validated, and tested on a dataset with labeled comments. Evaluation metrics like accuracy, precision, recall, and F1 score are used to measure the performance of each model.

## Table of Contents
- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [BERT Model](#bert-model)
  - [Training and Evaluation](#training-and-evaluation)
- [Custom GRU with Attention Model](#custom-gru-with-attention-model)
  - [Training and Evaluation](#training-and-evaluation-1)
- [Results](#results)
- [Future Enhancements](#future-enhancements)

## Requirements
- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- scikit-learn
- tqdm
- pandas
- numpy

## Dataset
The dataset used consists of labeled text comments with two categories. Each comment is tokenized, padded, and fed into the models for training. The dataset should have two columns:

- comment: The text sequence (comment).
- label: The corresponding label (0 or 1).

## BERT Model
In the first part of the project, we leverage the pre-trained BERT model for sequence classification. The BERT tokenizer is used to convert the text into tokens, and the BertForSequenceClassification model is fine-tuned on the dataset.

### Training and Evaluation
- Tokenizer: BERT tokenizer with max sequence length of 32.
- Model: bert-base-cased with 2 output labels.
- Optimizer: AdamW optimizer.
- Loss Function: Cross-entropy loss (included in BERT's BertForSequenceClassification).
- Training: The model is trained for 5 epochs with a batch size of 512.
- Metrics: Accuracy, Precision, Recall, and F1 Score.

During training, the model is evaluated on a validation set, and once trained, it is tested on a separate test set.

## Custom GRU with Attention Model
In the second part of the project, we build a custom GRU model with an attention mechanism to enhance the performance of the sequence classification task.
- Embeddings: Word-level embeddings generated from the vocabulary of the dataset.
- GRU: A single-layer GRU to process the sequences.
- Attention Mechanism: Helps focus on important words in the sequence by calculating attention weights.
- Final Layer: A fully connected layer followed by a sigmoid activation for binary classification.
### Training and Evaluation
- Word Tokenization: A custom tokenizer is used to convert the comments into a list of word indices, which is padded or truncated to a fixed length.
- Model: The GRU with Attention is built using PyTorch.
- Optimizer: AdamW optimizer with a learning rate of 1e-5.
- Loss Function: Binary Cross Entropy with Logits (BCEWithLogitsLoss).
- Training: The model is trained for 5 epochs with a batch size of 512.
- Metrics: Accuracy, Precision, Recall, and F1 Score, similar to the BERT model.

## Results
After training and evaluating both models on the test dataset, the performance metrics are compared.

### Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1 Score

Here are the sample results (you can update with your actual results):
|Model	            |Accuracy|Precision|Recall|F1 Score
|-------------------|--------|---------|------|--------
|BERT-based Model	  |0.7446  |0.7472   |0.7446|0.7453
|GRU with Attention	|0.8147  |0.6637   |0.8147|0.7315

## Future Enhancements
In future iterations of this project, several enhancements can be made:

- Hyperparameter Tuning: Experiment with different learning rates, batch sizes, and model architectures.
- Additional Layers: Add more GRU layers or explore different types of recurrent networks (e.g., LSTMs).
- Data Augmentation: Use data augmentation techniques to improve model robustness.
- Multiclass Classification: Extend the model to handle multiclass classification tasks.
- Advanced Pretrained Models: Experiment with larger transformer models such as RoBERTa or GPT-based models.
