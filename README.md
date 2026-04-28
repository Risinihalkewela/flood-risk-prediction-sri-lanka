# Flood Risk Prediction in Sri Lanka 🌊

This project applies **machine learning classification algorithms** to predict flood occurrences in Sri Lanka using environmental and geographic data. The study follows the **CRISP-DM methodology** (Business Understanding, Data Understanding, Data Preparation, Modelling, Evaluation, Deployment).

## 📊 Dataset
- ~25,000 geo-tagged records across Sri Lanka
- Features: rainfall, elevation, distance to rivers, land cover, soil type, vegetation indices, population density, infrastructure indicators
- Target: flood occurrence (binary classification)

## ⚙️ Methodology
- **Exploratory Data Analysis (EDA):** histograms, boxplots, correlation heatmaps
- **Data Preparation:** missing value handling, encoding categorical variables, scaling, class imbalance adjustment
- **Models Implemented:**
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting

## 🏆 Results
- **Gradient Boosting:** Accuracy = 91.42%, ROC-AUC = 0.9568
- **Decision Tree:** Highest Recall = 0.8927 (best for early warning)
- **Final Model:** Decision Tree chosen via composite scoring (accuracy + recall + ROC-AUC)

## 🔑 Key Insights
- Flood detection improves at lower classification thresholds (early warning advantage).
- Environmental features like rainfall intensity, elevation, and land cover strongly influence flood risk.
- Machine learning can support **disaster preparedness** and **risk management** in Sri Lanka.

## 🛠️ Tech Stack
- Python, Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn
- CRISP-DM Framework


✨ *This project demonstrates how machine learning can enhance flood risk prediction and disaster management in Sri Lanka.*
