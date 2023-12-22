# Machine Learning Project: Using Supervised Learning to predict who will catch the Smith Parasite

This project was developed for the Machine Learning course in the Master's degree in Data Science and Advanced Analytics at NOVA IMS.

The goal of this project was to use supervised learning algorithms in order to do binary classification (will catch it, will not) using data describing (fictional) people's health habits, personal characteristics and answers from a survey regarding mental and physical health. The end goal was to use the best model to do final predictions on unlabeled data (the f1 score would later be released from the kaggle private competition). All code was developed in Python.

## Main Steps (and main methods/algorithms)

- Data exploration with checks for missing data, inconsistencies and outliers (boxplots, histograms, DBScan)
- Preprocessing with inconsistencies, outliers and missing values correction, feature engineering, encoding, imputing and scaling; we built a custom pipeline to do all of this, as we later meant to do cross-validation, so were very careful about data leakage so we weren't mistaken by the results in the model evaluation part (RFE, column transformer, Pipeline, kNNImputer, Ordinal and OneHot Encoders, MinMaxScaling, Decision Trees, Lasso Regression, Chi-Squared test, Mutual Information)
- Modelling with Cross Validation (Random Forest, Extremely Randomized Trees, Gradient Boosting, SVM, kNN, MLP, Bagging and Voting); hyperparameter tuning with Randomized Search
- Models' results assessment and choice (CV f1 score on test set)
- Final Predictions.

Note that the notebook has everything which is not relevant to the final solution commented, so one can run the final solution without having to wait as much for other models, exploration and particularly the hyperparameter tuning process to run.
