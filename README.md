# 🍷TrueTaste: AI Wine Grader

### 🚀 Live Demo: https://wine-quality-predictor-jenycbgxnr5obxamxspukb.streamlit.app/

### 📖 The Story
In a winery producing thousands of gallons every year, grading is currently done by human tasters. But humans get tired, and taste buds are subjective. A "Standard" batch accidentally labeled as "Premium" can ruin a 50-year brand reputation.
This project introduces a Digital Sommelier. By analyzing the chemical "DNA" of wine, this AI-driven system provides objective, consistent grading to help wineries maximize revenue and protect their brand legacy.

### 🌟 Key Features
- Chemical DNA Analysis: Uses 12+ chemical markers (Acidity, Alcohol, Sulphates) to grade wine.
- Predictive Pricing Engine: Automatically estimates market value ($15 vs $65) based on quality.
- Model Intelligence Hub: Real-time validation using Random Forest with an AUC-ROC of 0.912.
- Synthetic Data Balancing: Implemented SMOTE to handle the rarity of "Premium" wine samples.

### 📊 The Data Science Workflow
1. Data Insights (EDA)
- We analyzed over 6,400 samples. Key discoveries included:
- The Alcohol Factor: Higher alcohol content is a strong indicator of "Premium" status.
- The Vinegar Factor: Volatile Acidity is a "deal-breaker"—high levels instantly drop quality.
- The Balance: Premium wine requires a specific "Goldilocks zone" of pH and density.

2. Machine Learning Pipeline
- Pre-processing: Scaling, Encoding, and Handling Imbalanced Classes using SMOTE.
- Algorithms Tested: Logistic Regression, KNN, Decision Trees, and Random Forest.
- Winning Model: Random Forest 🌲
  - Accuracy: 88.5%
  - AUC-ROC: 91.2%
  - Precision: 79.3% (Prioritizing brand safety over false positives).

### 🛠️ Tech Stack
- Language: Python
- Libraries: Pandas, NumPy, Scikit-Learn, Plotly
- Deployment: Streamlit
- Techniques: SMOTE, Feature Importance, Bivariate/Univariate Analysis

### 🏗️ Installation & Setup
1. Clone the repository: git clone https://github.com/your-username/wine-quality-ai.git
2. Install dependencies: pip install -r requirements.txt
3. Run the app: streamlit run app.py

### 💡 Business Value (The "Why")
Winery Owners: Identify "hidden gems" in the cellar to increase profit margins.

Quality Engineers: Reduce manual testing time by 80% via automated pre-screening.

Consumers: Guaranteed consistency in every bottle purchased.

🛤️ Roadmap
- [ ] Implement Hyperparameter Tuning (GridSearchCV) to further boost F1-Score.
- [ ] Add "Region-based" pricing logic for global winery support.
- [ ] Integration with IoT sensors for real-time fermentation monitoring.

Deployment: Streamlit

Techniques: SMOTE, Feature Importance, Bivariate/Univariate Analysis
