# 📈 CreatorPulse: AI-Powered YouTube Content & Performance Assistant

> **Business Analytics Research Project:** A predictive intelligence and content-generation suite that helps YouTube creators optimize video titles, tags, and posting schedules before publishing.

---

## 📌 Overview

Thousands of creators rely on guesswork for critical meta-data decisions. **CreatorPulse** bridges the gap between raw YouTube analytics and actionable content strategy. 

Trained on 77,596 trending YouTube videos from India and 18,644 viewer comments, this tool provides an end-to-end planning pipeline:
1. **Predictive Analytics:** Random Forest models forecast engagement rates and classify ideas as high-performers (top 25% engagement).
2. **Audience Sentiment:** VADER-based NLP analyzes viewer comments to extract audience desires (e.g., requests for follow-ups, complaints about costs).
3. **AI Agent Loop:** An integrated Large Language Model (LLM) generates titles, hooks, and hashtags, which are then scored by our ML model and iteratively improved in a "Self-Refine" loop.
4. **Creator Dashboard:** An interactive Streamlit app providing optimal posting times, trending topic analysis, and content scoring.

---

## 📸 Application Screenshots

<img width="903" height="419" alt="Screenshot 2026-10-18 202818" src="https://github.com/user-attachments/assets/5f519497-e901-4901-9d24-a1e0ce6515c0" />
<img width="883" height="457" alt="Screenshot 2026-10-18 202906" src="https://github.com/user-attachments/assets/8e2e08b3-fc64-41e9-8dad-ded12d40a191" />
<img width="901" height="401" alt="Screenshot 2026-10-18 203245" src="https://github.com/user-attachments/assets/7943543e-fb68-49fd-8100-7b5ebee27f67" />
<img width="897" height="448" alt="Screenshot 2026-10-18 202936" src="https://github.com/user-attachments/assets/d451e346-b9dd-4b26-ace4-ca62e348e0f0" />
<img width="897" height="449" alt="Screenshot 2026-10-18 203131" src="https://github.com/user-attachments/assets/fe3efef8-fbb9-45e5-b46f-acd184e5a73b" />


## 🧠 System Architecture & "Generate-Score-Improve" Loop

CreatorPulse operates using a custom agentic loop inspired by Self-Refine methodologies:

```text
 ┌────────────────┐     1. Generates 5 variations       ┌─────────────────────┐
 │ Creator Idea,  │ ──► (Titles, Captions, Hooks) ───►  │ LLM Content Engine  │
 │ Category, Time │                                     └─────────┬───────────┘
 └────────────────┘                                               │ 2. Scores
                                                                  ▼
 ┌────────────────┐     4. Outputs final optimized      ┌─────────────────────┐
 │  Final Ranked  │ ◄── content (If score stops   ◄───  │ Random Forest Model │
 │   Variations   │     improving)                      │ (Predicts Success)  │
 └────────────────┘                                     └─────────┬───────────┘
         ▲                                                        │ 3. Feeds back
         └────────────────── (Repeats up to 3 Rounds) ────────────┘
```

### Core Machine Learning Modules
* **High-Performer Classifier:** Random Forest (150 trees, max depth 16) predicting if a video will hit the top 25% engagement rate. 
* **Sentiment Analyzer:** VADER rule-based model analyzing comment polarity.
* **Views & Engagement Regressors:** Secondary Random Forest models estimating raw views and engagement ratios based on publish hour, title length, and tag counts.

---

## 📊 Experimental Results & ROI

The system was rigorously evaluated on held-out test data (20% split) and through an automated A/B test simulation on 30 fresh content ideas across 10 categories.

### Model Performance
| Model / Task | Evaluation Metric | Result |
| :--- | :--- | :---: |
| **High-Performer Classifier** | ROC-AUC | **0.837** |
| **High-Performer Classifier** | Accuracy | **77.5%** |
| **Sentiment Analysis (VADER)** | Accuracy (vs. 62.2% baseline) | **71.7%** |
| **Engagement Regressor** | R-squared | **0.385** |

### Content Generation ROI (Business Value)
When testing the AI loop against 30 original creator ideas:
* The LLM's optimized title scored higher than the original idea in **87% of cases** (26 of 30).
* The average uplift in the high-performer probability score was **+5.4%** (with a peak improvement of +23.0%).

---

## 📂 Repository File Structure

```text
CreatorPulse_AI_Assistant/
│
├── 03_Notebooks/                # EDA, Sentiment Analysis, and Random Forest training
├── 04_Dashboard/                # Analytics dashboard components and visualizations
├── 05_App_LLM/                  # Streamlit app, API integrations, and agent loop logic
│
├── requirements.txt             # Environment dependencies
├── LICENSE                      # MIT Open Source License
└── README.md                    # Project documentation
```

---

## 🗄️ Datasets & Pre-Trained Model Checkpoints

To comply with repository size limits, the raw datasets (trending videos & comments) and heavy Random Forest `.pkl` weights are hosted securely via Google Drive.

* 📥 **[Access Full Project Data & Trained Models (Google Drive)](https://drive.google.com/drive/folders/1GE-D8CRiRGyLZIvoFDwCUZ11fYp2TaiX?usp=sharing)**
  * *Note: The data represents Indian trending videos from approximately two years prior to project completion. Predictive scores are estimates based on trending behaviors, not guaranteed real-world views.*

---

## 👥 Authors & Collaborators

* **Srushti Lambdade** (PRN: 25030421021) 
* **Sanobar Shaikh** (PRN: 25030421039) 
* **Prajakta Suryavanshi** (PRN: 25030421029) 

**B.Sc. Artificial Intelligence – Symbiosis Artificial Intelligence Institute (SAII, SIU Pune)**

### Mentorship & Guidance
* **Internal Guides:** Dr. Bela Joglekar, Dr. Pradnya Patil, Dr. Sahas V Swamy
* **External Guides (Industry):** Mr. Sanchit Mehrotra & Mr. Chetan Bais (Co-Founders, Zeeplework)

---

## ⚠️ Disclaimer
This tool is built on trending video datasets, which inherently introduces success bias. Predictions are optimistic estimates and should be used for strategic planning rather than guaranteed outcome forecasting.
