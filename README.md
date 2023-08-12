# Graduates-Admission-Prediction-Using-DL


This is a machine learning project that predicts the chance of admission for graduate students based on various features such as GRE Score, TOEFL Score, University Rating, Statement of Purpose (SOP), Letter of Recommendation (LOR), CGPA, and Research experience. The project includes data preprocessing, model training using TensorFlow/Keras, and a Streamlit web application for prediction.

## Project Overview

This project aims to predict the likelihood of a student's admission to a graduate program based on their academic and research profile. It includes the following steps:

1. *Data Loading and Preprocessing*: The project starts by loading the dataset and splitting it into features (X) and target (y). The features are then scaled using `MinMaxScaler` to prepare them for model training.

2. *Model Creation*: A neural network model is built using TensorFlow/Keras. The model consists of three hidden layers with ReLU activation functions and a linear output layer. It is compiled with the mean squared error loss function and the Adam optimizer.

3. *Model Training*: The model is trained on the scaled training data using the `fit` function. The training data is split into training and validation sets for monitoring model performance during training.

4. *Model Evaluation*: The trained model is evaluated using the R-squared metric (`r2_score`) on the test data. This metric provides insight into how well the model's predictions align with the actual target values.

5. *Streamlit Web Application*: The final trained model is used to create a Streamlit web application. Users can input their academic and research details, and the application provides a prediction of their chance of admission.

## How to Use

1. Clone this repository to your local machine:

```bash
git clone https://Omkar-Rajkumar-Khade/Graduates-Admission-Prediction-Using-DL.git
```
2. Install the required Python packages:
```bash
pip install streamlit pandas numpy scikit-learn tensorflow
```
3. Run the Streamlit app:
```bash
streamlit run app.py
```
4. Open the provided Streamlit URL in your web browser. You will see an interface where you can input the necessary information.

## Dataset
The original dataset used for training the model can be found at Kaggle:https://www.kaggle.com/datasets/mohansacharya/graduate-admissions

The dataset contains several parameters which are considered important during the application for Masters Programs.
The parameters included are :

1. GRE Scores ( out of 340 )
2. TOEFL Scores ( out of 120 )
3. University Rating ( out of 5 )
4. Statement of Purpose and Letter of Recommendation Strength ( out of 5 )
5. Undergraduate GPA ( out of 10 )
6. Research Experience ( either 0 or 1 )
7. Chance of Admit ( ranging from 0 to 1 )


