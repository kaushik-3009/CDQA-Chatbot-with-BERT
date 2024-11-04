# Closed-Domain Passage-Based Question Answering System using NLP and Transfer Learning (BERT)

## Abstract

This project aims to develop a robust closed-domain question answering system using Bidirectional Encoder Representations from Transformers (BERT) and transfer learning. The system leverages the pre-trained BERT model, fine-tuned on the SQuAD dataset, to accurately answer questions based on given passages of text.  We explore the effectiveness of BERT and transfer learning in enhancing question answering accuracy and discuss potential improvements through further hyperparameter tuning.

## Introduction

Efficiently extracting information from long-form text is crucial. Passage-based question answering models, combined with information retrieval, offer a solution.  This project utilizes transfer learning, specifically with the BERT architecture, to improve performance and reduce training time by leveraging pre-existing knowledge from massive text datasets. We use the SQuAD dataset for training and evaluation.

## Proposed System

The system follows the KDD process:

1. **Dataset Selection:** Stanford Question Answering Dataset (SQuAD).
2. **Preprocessing:** Cleaning, formatting for BERT (using `simpletransformers`).
3. **Exploratory Data Analysis:** Wordcloud, Word2Vec embeddings.
4. **Model:** Fine-tuned BERT model (`bert-base-cased`).
5. **Implementation:** Python, Google Colab, `transformers` library.

## Architecture

The system uses the BERT architecture for question answering. Input text (question and passage) is tokenized, embedded, and passed through transformer layers. Start and end vectors are generated to predict the answer span within the passage.

## Usage

1. **Installation:** `pip install transformers simpletransformers` (and other dependencies listed in the report).
2. **Data Preparation:**  Download and preprocess the SQuAD dataset.
3. **Model Training:** Fine-tune the BERT model using the provided BERT script.
4. **Inference:**  Use the trained model in W2V script to answer questions given a passage.


## Results

The model achieved promising results on the SQuAD dataset and show potential capabilities which can be used reliably in industries like Healthcare, Finance, and Education to provide accurate results.

## Future Enhancements

* Domain-specific fine-tuning.
* Multi-task learning (e.g., question answering, summarization).
* Integration of external knowledge sources.
