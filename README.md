Overview
This project implements advanced machine learning and natural language processing techniques for automated fact verification using the Triple-R dataset. The goal is to predict the truthfulness of real-world claims based on external evidence and to provide human-readable explanations for each decision, enhancing transparency and interpretability in automated fact-checking systems.

Dataset Description
Triple-R: Automatic Reasoning for Fact Verification
The Triple-R dataset is designed to improve fact verification by combining external evidence retrieval and the generation of natural language explanations. Built upon the LIAR dataset, Triple-R introduces a three-stage pipeline:

Retriever: Gathers relevant evidence from the web for each claim.

Ranker: Scores and selects the most pertinent paragraphs from the retrieved documents.

Reasoner: Uses GPT-3.5-Turbo to generate human-readable explanations for each claim based on the selected evidence.

Each data sample includes:

id: Unique identifier

statement: The claim to be verified

label: Truthfulness of the claim (e.g., true, false, half-true, pants-fire, etc.)

evidence: Relevant information retrieved from the web

reason: (Train set only) AI-generated explanation for the claim

The dataset is ideal for training models in misinformation detection and explainable AI, supporting applications that require transparent and interpretable decision-making.

Project Objectives
Fact Verification: Predict the truthfulness label for each claim using both the claim and its supporting evidence.

Explainable AI: Leverage and evaluate the generated explanations to enhance model transparency.

Model Comparison: Benchmark classical and modern NLP models, including:

TF-IDF + Logistic Regression (baseline)

Sentence Transformers + Logistic Regression

Fine-tuned BERT transformer models

Approach
1. Data Preparation
Loaded and preprocessed the Triple-R dataset, combining claims and evidence for model input.

Explored label distribution and evidence types to understand dataset complexity.

2. Baseline Modeling
Implemented TF-IDF vectorization with Logistic Regression as a baseline.

Evaluated performance using accuracy, F1-score, and confusion matrix.

3. Advanced Modeling
Used Sentence Transformers (all-mpnet-base-v2) to generate dense semantic embeddings for claims and evidence.

Trained a Logistic Regression classifier on these embeddings for improved accuracy.

Fine-tuned a BERT transformer model for direct sequence classification, leveraging HuggingFace’s Transformers library.

4. Class Imbalance Handling
Applied class weighting in model training to address imbalanced label distribution.

5. Evaluation
Compared models on test accuracy, F1-score, and per-class performance.

Visualized confusion matrices for detailed error analysis.

Results
Model	Test Accuracy	Weighted F1 Score
TF-IDF + Logistic Regression	~29%	~28%
Sentence Transformers + Logistic Reg	~42%	~41%
Fine-tuned BERT	~52%	~51%
Sentence Transformers and BERT models significantly outperformed the baseline, demonstrating the importance of semantic understanding in fact verification.

Class weighting and evidence-aware input formatting further improved performance, especially for minority classes like "pants-fire".

How to Run
Clone this repository and upload train.json and test.json from the Triple-R dataset.

Install dependencies:

bash
```
pip install -r requirements.txt
```
Run the main notebook in Google Colab or locally:

TripleR_Fact_Verification.ipynb

Key Files
TripleR_Fact_Verification.ipynb — Main notebook with all experiments and results.

train.json, test.json — Triple-R dataset files (not included; download from UNB CIC).

requirements.txt — List of Python dependencies.

References
Triple-R Dataset at UNB CIC

Triple-R: Automatic Reasoning for Fact Verification Using Language Models (Kanaani et al., 2024)
