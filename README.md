# House Price Prediction

## Project Overview
This notebook project builds a regression model to predict house sale prices using the Kaggle House Prices dataset. The analysis includes exploratory data checks, missing value handling, categorical encoding, XGBoost model training, evaluation, and submission generation.

## What the code does
- Loads training data from `data/train.csv`
- Explores data shape, missing values, and price distribution
- Performs feature analysis and correlation inspection
- Cleans missing values using domain-aware rules
  - fills `None` for absent categorical features
  - fills `0` for missing numeric basement and garage measurements
  - imputes `LotFrontage` using neighborhood medians
  - fills remaining missing values with median or mode
- Encodes categorical variables using label encoding
- Trains an `XGBRegressor` on log-transformed sale prices
- Evaluates the model on a holdout test split
- Produces predictions for `data/test.csv` and saves submission to `output/house_price_submission.csv`

## Evaluation Summary
The notebook evaluates model performance using the following metrics on the held-out test split:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² score

It also visualizes:
- the original and log-transformed sale price distribution
- actual vs predicted prices
- top feature importances

## Notes on code quality
The current code pipeline is well structured for an exploratory modeling workflow, and it demonstrates:
- clear preprocessing logic with domain-specific missing value treatment
- consistent treatment of test data using the same cleaning rules as training data
- correct use of log transformation for the target variable
- proper separation of features and target, followed by a train/test split

### Improvements to consider
- use consistent encoding for test data by fitting the encoder on training data only
- validate that the test dataset has the same columns as the training features before prediction
- add cross-validation or hyperparameter tuning for more robust model selection
- use a dedicated preprocessing pipeline or function structure for readability and reuse

## Files
- `notebooks/house_price_analysis.ipynb` — main analysis notebook
- `data/train.csv` — training dataset
- `data/test.csv` — test dataset for submission
- `output/house_price_submission.csv` — generated submission file
- `output/price_distribution.png` — sale price distribution plot
- `output/correlation_heatmap.png` — feature correlation heatmap
- `output/predictions.png` — actual vs predicted price plot

## How to run
1. Open `notebooks/house_price_analysis.ipynb` in Jupyter or VS Code.
2. Run the notebook cells sequentially.
3. The final submission file is saved to `output/house_price_submission.csv`.

## Conclusion
This project provides a solid baseline for house price prediction. The notebook contains clear data cleaning and model evaluation steps, and it can be extended with improved feature engineering, encoder handling, and model tuning for better performance.
