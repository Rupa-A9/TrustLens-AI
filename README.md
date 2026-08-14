# 🛡️ TrustLens AI

<p align="center">
  <img src="https://img.shields.io/badge/Pythonx-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Machine%20Learning-orange?style=for-the-badge" alt="Linear SVM">
</p>

<h3 align="center">
  AI-Powered Fake Review Detection
</h3>

<p align="center">
  Analyze product reviews, detect potentially suspicious content,
  understand model predictions, and generate downloadable reports.
</p>

<p align="center">
  <a href="https://trustlens-ai-rupa.streamlit.app/">
    <img src="https://img.shields.io/badge/OPEN%20LIVE%20APPLICATION-1E3A8A?style=for-the-badge">
  </a>
</p>

---

## About The Project

**🛡️ TrustLens AI** is a machine-learning powered web application designed
to identify potentially fake or suspicious product reviews.

The system uses:

- Natural Language Processing
- TF-IDF Vectorization
- Linear Support Vector Machine
- Python
- Scikit-learn
- Streamlit

Users can analyze individual reviews, process multiple reviews through CSV
files, view model performance, understand the machine-learning pipeline,
and generate downloadable PDF analysis reports.

---

## Problem Statement

Online product reviews strongly influence purchasing decisions.

However, fake or misleading reviews can:

- Manipulate product ratings
- Mislead customers
- Reduce trust in online marketplaces
- Make genuine products harder to evaluate

TrustLens AI attempts to address this problem by learning textual patterns
from labeled review data and using those patterns to classify new reviews.

---

## Solution

TrustLens AI processes review text through a machine-learning pipeline:

```text
                 ┌─────────────────────┐
                 │   Product Review    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Text Preprocessing  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  TF-IDF Vectorizer  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    Linear SVM       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │     Prediction      │
                 └──────────┬──────────┘
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
          Genuine Review       Fake / Suspicious
````
```
---

## Key Features

| Feature                 | Description                                  |
| ----------------------- | -------------------------------------------- |
| Single Review Detection | Analyze an individual product review         |
| Batch Detection         | Upload a CSV and analyze multiple reviews    |
| TF-IDF                  | Convert review text into numerical features  |
| Linear SVM              | Classify reviews using the trained model     |
| Model Metrics           |View evaluation results and model information             |
| PDF Reports             | Download review analysis reports             |
| Review Statistics       | Characters, words, lines and processing time |
| Fast Prediction         | Real-time review classification              |
| Cloud Deployment        | Deployed using Streamlit Community Cloud     |

---

## Model Performance

The trained Linear SVM model achieved the following evaluation results:

<div align="center">

| Metric     |         Result |
| ---------- | -------------: |
| Accuracy   |     **89.16%** |
| Precision  |     **89.05%** |
| Recall     |     **89.29%** |
| F1 Score   |     **89.17%** |
| Best Model | **Linear SVM** |

</div>

> These values represent evaluation results from the trained model.
> Individual predictions may still be incorrect.

---

## Machine Learning Pipeline

### 01 — Natural Language Processing

The application receives human-written product review text.

Example:

```text
"The battery lasts all day and the sound quality is excellent."
```

The text is processed before being passed to the machine-learning model.

---

### 02 — TF-IDF Vectorization

TF-IDF converts text into numerical feature vectors.

#### TF — Term Frequency

Measures how frequently a word occurs within a particular review.

#### IDF — Inverse Document Frequency

Measures how informative a word is across the review dataset.

The resulting representation is:

```text
Review Text
     ↓
TF-IDF Vectorization
     ↓
Numerical Feature Vector
```

---

### 03 — Linear SVM

The TF-IDF feature vector is passed to the trained Linear Support Vector
Machine classifier.

```text
TF-IDF Features
       ↓
Linear SVM
       ↓
Classification
```

The model returns the predicted review category.

---

## Single Review Detection

The **Detect** page allows users to enter a product review and analyze it
in real time.

The application displays:

* Prediction
* Prediction confidence
* Character count
* Word count
* Line count
* Processing time
* Model information
* Analyzed review
* Downloadable PDF report

### Workflow

```text
Enter Review
     ↓
Analyze Review
     ↓
Text Preprocessing
     ↓
TF-IDF
     ↓
Linear SVM
     ↓
Prediction
     ↓
Generate Report
```

---

## Batch Detection

The **Batch Detection** page allows users to upload a CSV containing
multiple product reviews.

The application processes the reviews using the same trained prediction
pipeline.

This allows multiple reviews to be analyzed efficiently.

---

## Model Metrics

The Model page provides a concise view of the machine-learning model
used by TrustLens AI.

It includes:

- Accuracy
- Precision
- Recall
- F1 Score
- Selected classifier
- Dataset size
- TF-IDF feature configuration
- Evaluation comparison
- Dataset preview
- Model configuration

The purpose of this page is to make the model evaluation and configuration
transparent without exposing unnecessary implementation details to the
application user.

---

## PDF Report Generation

After analyzing a review, TrustLens AI can generate a downloadable PDF
report.

The report can contain:

* Review text
* Prediction
* Confidence
* Review statistics
* Processing information
* Model information
* TF-IDF information
* Linear SVM information

PDF generation is implemented using **ReportLab**.

---

## Exploratory Data Analysis

The project contains an EDA pipeline for analyzing the review dataset.

The analysis includes:

* Dataset shape
* Dataset columns
* Data types
* Missing values
* Label distribution
* Rating distribution
* Product categories
* Review length
* Word cloud analysis

---

## Technology Stack

### Programming

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square\&logo=python\&logoColor=white)

### Web Application

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square\&logo=streamlit\&logoColor=white)

### Machine Learning

![Scikit Learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square\&logo=scikit-learn\&logoColor=white)

![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square\&logo=numpy\&logoColor=white)

![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square\&logo=pandas\&logoColor=white)

### NLP

![NLTK](https://img.shields.io/badge/NLTK-154F3C?style=flat-square)

![TF-IDF](https://img.shields.io/badge/TF--IDF-Text%20Vectorization-blue?style=flat-square)

### Visualization

![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)

![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square\&logo=plotly\&logoColor=white)

### PDF Reporting

![ReportLab](https://img.shields.io/badge/ReportLab-PDF-red?style=flat-square)

---

## Project Structure

```text
TrustLens-AI/
│
├── .streamlit/
│   └── config.toml
│
├── assets/
│   └── css/
│       ├── components.css
│       ├── globals.css
│       ├── pages.css
│       └── style.css
│
├── components/
│   ├── hero.html
│   ├── hero.py
│   ├── navbar.html
│   ├── navbar.py
|
├── data/
│   └── processed review data
│
├── models/
│   ├── model.pkl
│   ├── vectorizer.pkl
│   ├── metrics.json
│   ├── train.py
│   ├── evaluate.py
│   └── compare_models.py
│
├── outputs/
│   └── generated analysis outputs
│
├── pages/
│   ├── home.py
│   ├── detect.py
│   ├── batch.py
│   ├── dashboard.py
│
├── services/
│   └── predictor.py
│
├── utils/
│   ├── eda.py
│   ├── load_css.py
│   ├── pdf.py
│   ├── preprocess.py
│
├── app.py
├── config.py
├── download_nltk.py
├── notebooks.txt
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Rupa-A9/TrustLens-AI.git
```

### 2. Enter the Project Directory

```bash
cd TrustLens-AI
```

### 3. Create a Virtual Environment

#### Windows

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
```

Activate:

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Locally

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will normally open at:

```text
http://localhost:8501
```

---

## Required Model Files

The prediction system requires:

```text
models/
├── model.pkl
└── vectorizer.pkl
```

`model.pkl` contains the trained Linear SVM classifier.

`vectorizer.pkl` contains the trained TF-IDF vectorizer used to transform
review text into the feature representation expected by the classifier.

---

## Deployment

TrustLens AI is deployed using **Streamlit Community Cloud**.

### Deployment Architecture

```text
                    GitHub Repository
                           │
                           ▼
                  ┌─────────────────┐
                  │   TrustLens AI  │
                  │    Project      │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │    Streamlit    │
                  │ Community Cloud │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   Live Web App  │
                  └─────────────────┘
```

---

## Complete Application Workflow

```text
                         ┌─────────────────┐
                         │  Product Review │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Text Processing │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │      TF-IDF     │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │    Linear SVM   │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │    Prediction   │
                         └────────┬────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
             Genuine Review             Fake / Suspicious
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                         ┌─────────────────┐
                         │   PDF Report    │
                         └─────────────────┘
```

---


## Author

### Siripurapu Rupasri - Rupa-A9

---

<p align="center">

<b>🛡️ TrustLens AI</b>

Analyze Reviews. Detect Deception. Build Trust.

Built with Python • NLP • TF-IDF • Linear SVM • Streamlit

</p>