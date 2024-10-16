# Diabetes Prediction System - Web App

## Overview
The **Diabetes Prediction System** is a web application designed to predict the likelihood of a person having diabetes based on various clinical factors. It leverages machine learning algorithms to provide healthcare professionals with a reliable and user-friendly tool to assist in diagnosing diabetes. The system incorporates advanced models, including **Random Forest**, **XGBoost**, and **K-Nearest Neighbors (KNN)**, to achieve highly accurate predictions.
<br> <br>For more detailed insights into the machine learning models used, including the specific algorithms, hyperparameter tuning processes, and evaluation metrics such as precision, recall, and F1 score, you can explore the full implementation and codebase in the repository below:
 <br> <br>[Diabetes Prediction System - GitHub Repository](https://github.com/Abishakm1507/Diabetes-Prediction)
 <br> <br>This repository provides a comprehensive breakdown of the data preprocessing steps, model training, and the evaluation techniques applied to achieve these results.

## Features
- **Machine Learning Models:** Utilizes an ensemble of machine learning algorithms (Random Forest, XGBoost, KNN) to enhance prediction accuracy.
- **Data Preprocessing:** Handles missing data, normalizes features, and encodes categorical variables for better model interpretation.
- **Hyperparameter Tuning:** Optimizes each model through techniques like **GridSearchCV** for better performance.
- **Voting Classifier:** Combines predictions from multiple models to improve overall accuracy.
- **Performance Evaluation:** Assesses the model's performance using metrics such as accuracy, precision, recall, and confusion matrix.
- **User-Friendly Interface:** Provides a simple interface for healthcare professionals to input clinical data and receive predictions.
- **Feature Importance:** Offers insight into which factors have the most significant impact on diabetes risk.

## Key Features Analyzed
- **Glucose Levels**
- **Blood Pressure**
- **Body Mass Index (BMI)**
- **Insulin Levels**
- **Age**
- **Diabetes Pedigree Function (Family History)**

## Machine Learning Workflow
1. **Data Preprocessing:**
   - Handle missing or inconsistent data.
   - Normalize features to ensure uniformity in scale.
   - Encode categorical variables.
   - Split data into training and testing sets.

2. **Model Training:**
   - **Random Forest:** Fine-tuned using GridSearchCV to find the optimal number of trees and maximum depth.
   - **XGBoost:** Hyperparameters like learning rate, number of estimators, and max depth are optimized.
   - **KNN:** The number of neighbors is carefully selected based on model performance.

3. **Ensemble Approach:**
   - The system combines all models into a **Voting Classifier** to produce a robust and accurate prediction.

4. **Performance Evaluation:**
   - Metrics like **Accuracy**, **Precision**, **Recall**, and **Confusion Matrix** are used to evaluate model performance on the test set.

## Technologies Used
- **Frontend:**
  - HTML, CSS, JavaScript
  - Responsive and user-friendly interface for input and results.
  
- **Backend:**
  - Python (Flask)
  - Machine Learning with **Scikit-learn** and **XGBoost**
  
## Installation Guide

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Abishakm1507/Diabetes-Prediction-Web-App.git
   
1. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate 


3. **Install Required Dependencies:**
   ```bash
   pip install -r requirements.txt

4. **Run the Web Application:**
   ```bash
   python app.py

5. **Access the Web Application:**  Open your browser and go to http://127.0.0.1:5000/

## How to Use the System

1. Input the patient's clinical data such as glucose level, BMI, age, etc.
2. Click on the "Predict" button.
3. The system will display the predicted result along with the associated metrics.
4. If applicable, the system will highlight which clinical factors contributed most to the prediction.

## Performance Metrics

- The model's accuracy is around **84.41%**, which ensures a reliable prediction rate for the likelihood of diabetes.

## Future Enhancements

- Add more advanced models to further improve accuracy.
- Expand the dataset to include more clinical variables.
- Deploy the application in the cloud (AWS, Heroku, etc.) for broader access.

## Contribution

We welcome contributions from the community! Feel free to fork the repository and submit pull requests.
