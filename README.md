# 🏥 Chatbot for Healthcare using NLP and Python

An AI-powered **symptom-analysis and healthcare chatbot** that predicts possible diseases based on user-reported symptoms using Natural Language Processing (NLP) and K-Nearest Neighbors (KNN) classification. Trained on a Kaggle dataset of 4000+ medical records across multiple diseases, validated on 500+ test cases with 75% diagnostic accuracy.

---

## 🚀 Features

- **Symptom Input via Natural Language** — Users describe symptoms in plain English; NLP pipeline tokenizes and maps to structured symptom vectors
- **Disease Prediction** — KNN classifier trained on 4000+ Kaggle records predicts top probable diseases
- **75% Diagnostic Accuracy** — Validated across 500+ test cases
- **Real-time Triaging** — Response latency reduced by 30% via optimized inference pipeline
- **Confidence Scoring** — Returns probability scores for each predicted disease
- **Multi-turn Conversation** — Chatbot asks follow-up questions to refine diagnosis
- **AI-enhanced Decision Making** — PyTorch-based model for complex symptom pattern recognition

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.8+ |
| NLP | NLTK, spaCy |
| ML Model | KNN (scikit-learn), PyTorch |
| Dataset | Kaggle Disease-Symptom Dataset (4000+ records) |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Interface | CLI / JSON API |

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Training Dataset | 4000+ Kaggle medical records |
| Test Cases Validated | 500+ |
| Diagnostic Accuracy | 75% |
| Response Latency Reduction | 30% (vs baseline) |
| Diseases Covered | 40+ |
| Symptom Features | 130+ |

---

## ⚙️ How to Run

### Prerequisites
- Python 3.8+
- pip

### Setup Steps

```bash
# 1. Clone the repository
git clone https://github.com/aravindkanna8/Chatbot-for-healthcare-using-NLP-and-Python.git
cd Chatbot-for-healthcare-using-NLP-and-Python

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 4. Run the chatbot
python chatbot.py
```

### Requirements.txt
```
nltk==3.7
scikit-learn==1.1.2
torch==1.13.0
pandas==1.5.0
numpy==1.23.0
spacy==3.4.0
matplotlib==3.6.0
```

---

## 🧠 How It Works

```
User Input (Natural Language)
        ↓
   NLP Pipeline
   (Tokenization → Stopword Removal → Lemmatization)
        ↓
  Symptom Vector (130+ binary features)
        ↓
   KNN Classifier (k=5, trained on 4000+ records)
        ↓
  Top-3 Disease Predictions + Confidence Scores
        ↓
  PyTorch Model (complex pattern refinement)
        ↓
   Final Diagnosis + Follow-up Questions
```

---

## 📁 Project Structure

```
Chatbot-for-healthcare-using-NLP-and-Python/
├── chatbot.py                  # Main chatbot entry point
├── model/
│   ├── train_knn.py            # KNN model training
│   ├── train_pytorch.py        # PyTorch model training
│   └── predict.py              # Inference pipeline
├── nlp/
│   ├── preprocessor.py         # Tokenization, lemmatization
│   └── symptom_extractor.py    # NLP → symptom vector mapping
├── data/
│   ├── dataset.csv             # Kaggle disease-symptom dataset
│   └── symptom_severity.csv    # Symptom weight mapping
├── utils/
│   └── evaluator.py            # Accuracy, F1 evaluation
├── requirements.txt
└── README.md
```

---

## 💡 Sample Interaction

```
🤖 HealthBot: Hi! Describe your symptoms.

👤 User: I have a high fever, body aches, and fatigue since 2 days.

🤖 HealthBot: I've identified the following symptoms:
   • Fever (high)
   • Body ache
   • Fatigue

   Top predictions:
   1. Influenza        — 82% confidence
   2. Dengue Fever     — 71% confidence
   3. Typhoid          — 58% confidence

   Do you also have a headache or chills? (yes/no)
```

---

## 🎯 Problem Statement

Patients often delay medical consultation due to lack of awareness of symptoms. This chatbot provides an accessible first-line AI triage tool that maps natural language symptom descriptions to probable diagnoses, helping users make faster, informed decisions about seeking medical care.

---

## 👨‍💻 Author

**AravindSamy Selvaraj**
- LinkedIn: [linkedin.com/in/aravind-samy-s](https://linkedin.com/in/aravind-samy-s)
- GitHub: [github.com/aravindkanna8](https://github.com/aravindkanna8)
