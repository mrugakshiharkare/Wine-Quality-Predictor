### 🚀 Live Demo: https://wine-quality-predictor-jenycbgxnr5obxamxspukb.streamlit.app/

# 🍷 TrueTaste: AI Wine Grader
> **A Machine Learning solution for objective quality assessment and chemical analysis.**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://wine-quality-predictor-jenycbgxnr5obxamxspukb.streamlit.app/)

---

## 📖 Project Overview
Determining wine quality is traditionally a subjective process prone to human error. This project leverages **Random Forest Classification** to provide a data-driven alternative. By analyzing 11 chemical features (Acidity, Residual Sugar, Chlorides, Alcohol, etc.), the model identifies the "Chemical DNA" that defines a "High Quality" wine.

---

## 📊 Model Performance & Evaluation
To ensure the model is robust and reliable, I evaluated it using industry-standard classification metrics. 

| Metric | Score | Business Significance |
| :--- | :--- | :--- |
| **Accuracy** | **88.5%** | High overall reliability for batch classification. |
| **AUC-ROC** | **91.2%** | Excellent ability to distinguish between quality classes. |
| **Precision** | **79.3%** | Low False Positive rate (Protects Brand Reputation). |
| **Recall** | **56.4%** | High ability to capture true high-quality samples. |
| **F1-Score** | **65.9%** | Balanced harmonic mean of Precision and Recall. |

> **💡 Key Insight:** Feature Importance analysis revealed that **Alcohol Content** and **Sulphates** are the primary drivers of quality perception in this dataset, while **Volatile Acidity** shows a strong inverse correlation with quality.

---

## 🛠️ Environment Setup & Execution
Follow these steps to run the interactive Streamlit dashboard on your local machine:
1. Clone the repository  
git clone [https://github.com/mrugakshiharkare/Wine-Quality-Predictor.git](https://github.com/mrugakshiharkare/Wine-Quality-Predictor.git)  
cd Wine-Quality-Predictor

2. Setup Virtual Environment  
python -m venv venv  
source venv/bin/activate  # Windows: venv\Scripts\activate  

3. Install Dependencies  
pip install -r requirements.txt

4. Launch the AI Sommelier  
streamlit run app.py

## 🚀 Future Roadmap & Improvements
I am currently scaling this project from a "Predictor" to an "Enterprise Solution" through the following phases:

#### 🟢 Phase 1: Model Optimization
- [ ] **Hyperparameter Tuning:** Implementing `GridSearchCV` to optimize Forest depth and estimators.
- [ ] **Ensemble Learning:** Testing **XGBoost** and **LightGBM** to compare performance with Random Forest.

#### 🟡 Phase 2: Advanced Data Engineering
- [ ] **SMOTE Implementation:** Balancing the dataset to improve minority class detection (Excellent/Poor wines).
- [ ] **Cross-Validation:** Using **Stratified K-Fold** to ensure model stability across different data splits.

#### 🔴 Phase 3: Deployment & MLOps
- [ ] **Dockerization:** Containerizing the application for consistent deployment.
- [ ] **MLflow Integration:** Implementing experiment tracking to monitor model versioning and performance drift.

---

## 🤝 Let's Connect!
I am actively looking for **Data Analyst and Machine Learning** opportunities where I can apply my analytical skills to solve real-world problems.

<p align="left">
<a href="https://linkedin.com/in/mrugakshi-harkare-ab4701251" target="blank"><img align="center" src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="Mrugakshi Harkare" /></a>
<a href="mailto:mrugakshiharkare2003@gmail.com" target="blank"><img align="center" src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="mrugakshiharkare2003" /></a>
<a href="https://github.com/mrugakshiharkare" target="blank"><img align="center" src="https://img.shields.io/badge/Portfolio-100000?style=for-the-badge&logo=github&logoColor=white" alt="mrugakshiharkare" /></a>
</p>

---
*Developed with a focus on data integrity and actionable Machine Learning.*

