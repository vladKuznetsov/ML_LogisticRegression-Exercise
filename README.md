ML_LogisticRegression exercise detailing steps of the process

This is a simple project to demonstrate supervised machine learning using LogisticRegression model for classification.

1. Create a LogisticRegression Model
2. Perform Data clean-up (remove NoNs, convert categorical features to one-hot values, and create ndarray) 
using Pandas and NumPy
3. Apply PCA to this model to reduce dimensionality of the feature set.
4. Tune the model:
    - Use cross-validation with 6-folds  
    - Apply GridSearch framework from Scikit-learn to this model: Parameter tuning is performed with exhaustive 
    grid search on tuples (PCA redaction to given number of parameters, random initial state) using accuracy, 
    precision, and recall estimators
5. Present results on stdout

Requirements:
Python 3.7; compatibility with other versions is not tested. Pandas, NumPy, Scikit-learn, Matplotlib.

Usage: python main_MLLogisticRegression.py

Input data: Change to full pathname of your file on line #62 in main_MLLogisticRegression.py
fileName = "/Volumes/DATA_1TB/Safary_Downloads/ml_data.csv"
