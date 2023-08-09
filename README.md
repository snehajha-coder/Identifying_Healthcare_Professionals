This code is a part of the competition held at TechGig hosted by DOCEREE MEDIA INDIA PRIVATE LIMITED

# Healthcare Professional Identification Model

Welcome to the Healthcare Professional Identification Model repository. This repository contains code and instructions on how to identify Healthcare Professionals (HCPs) and predict their specialization using ad server logs. This project aims to build a robust model that can accurately classify users into the HCP category and determine their specialization based on available data.

## Objective

The main objective of this project is to develop a machine learning model that accurately predicts whether a user belongs to the HCP category and identifies their specialization based on various features extracted from ad server logs.

## Getting Started

To get started with the project, follow these steps:

1. **Setup Environment**: Ensure you have Python 3.x installed along with the required libraries such as pandas, numpy, matplotlib, tqdm, scikit-learn, and more. You can install these dependencies using the following command:

```bash
pip install pandas numpy matplotlib tqdm scikit-learn
```

2. **Data**: The project uses the following datasets:
   - `Doceree_HCP_Sample_Submission.csv`: Sample submission format
   - `Doceree_HCP_Train.csv`: Training dataset
   - `Doceree_HCP_Test.csv`: Testing dataset

3. **Code**: The project code is organized as follows:
   - Import necessary libraries and ignore warning messages.
   - Load and preprocess the data, converting some columns to categorical.
   - Feature engineering: Extract keywords from 'KEYWORDS' column, create new columns for each unique keyword, and assign values based on their presence.
   - Train and evaluate a Logistic Regression model for classification:
     - Split the data into training and testing sets.
     - Create and train a Logistic Regression model.
     - Calculate accuracy and generate a classification report.
   - Train and evaluate a Random Forest Classifier model for classification:
     - Split the data into training and testing sets.
     - Create and train a Random Forest Classifier model.
     - Calculate accuracy and generate a classification report.
   - Use the trained Random Forest model to predict HCP categories for the testing dataset.

4. **Execution**: Follow the code step by step, ensuring you have the required datasets in the same directory. Execute the code blocks to preprocess data, train models, and make predictions.

## Results

The models have been trained and evaluated using the provided datasets. Here are the results for both models:

### Logistic Regression Model
- Accuracy: 94.3%
- Classification Report:
  ```
              precision    recall  f1-score   support

         0.0       0.95      0.96      0.96     15981
         1.0       0.91      0.89      0.90      6807

    accuracy                           0.94     22788
   macro avg       0.93      0.93      0.93     22788
weighted avg       0.94      0.94      0.94     22788
  ```

### Random Forest Classifier Model
- Accuracy: 99.1%
- Classification Report:
  ```
              precision    recall  f1-score   support

         0.0       0.99      1.00      0.99     15874
         1.0       1.00      0.97      0.98      6914

    accuracy                           0.99     22788
   macro avg       0.99      0.99      0.99     22788
weighted avg       0.99      0.99      0.99     22788
  ```

## Conclusion

This project demonstrates the process of identifying Healthcare Professionals and their specializations using machine learning models. The achieved accuracy of the models showcases their effectiveness in classifying users based on the provided ad server logs. The trained Random Forest Classifier model, in particular, has shown outstanding performance with a 99.1% accuracy rate.

For any inquiries or collaboration opportunities, please contact Sneha Jha at jhasneha205@gamil.com



