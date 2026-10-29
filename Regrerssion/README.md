# House Price Prediction (Regression)

A supervised machine learning project that predicts a property's `TargetPrice` from tabular property and neighborhood attributes. The notebook explores the data, compares regression models with cross-validation, tunes a decision tree, evaluates errors, and creates predictions for an unlabeled competition test set.



## Business problem

Property valuation teams, lenders, real estate platforms, and prospective buyers may need an estimate of a property's value from available property and neighborhood information. A regression model can provide a consistent first estimate to help prioritize manual review, compare listings, or identify cases that merit closer attention.

This notebook frames the task as **regression**: predict the continuous `TargetPrice` field from the available features. The model is not an appraisal, does not establish causal drivers of price, and should not be used for financial decisions without appropriate validation and oversight.


The notebook uses the **House Price Prediction Challenge Regression** dataset from Kaggle, read from `estate_train.csv` and `estate_test.csv`.

The saved notebook reports:

- Training data: **16,512 rows and 12 columns**.
- Test data: **4,128 rows and 11 columns**.
- `TargetPrice` is available in the labeled training data and absent from the test data.
- `PropertyID` is retained for the prediction submission but excluded from the model features.
- The training columns include income level, property age, room and bedroom counts, neighborhood population, average occupancy, latitude, longitude, and derived room/bedroom ratios.

The test set is used to demonstrate the prediction and submission-file workflow. The notebook does not report a known-label score for that test set.

## Data setup

The notebook currently reads the files from Kaggle-specific paths. Download the files from the original source and either preserve those paths in Kaggle or change the two `read_csv` calls to point to your local files. A local project layout could be:

```text
project/
├── README.md
├── regression.ipynb
└── data/
    ├── estate_train.csv
    └── estate_test.csv
```

Do not redistribute the dataset unless its license and competition rules allow it. No data license or project license is specified in the notebook.
## Machine learning lifecycle

1. **Problem definition:** estimate a continuous property price from property and neighborhood attributes.
2. **Dataset selection:** use the provided labeled training dataset and separate unlabeled test dataset.
3. **Data inspection and EDA:** examine shapes, data types, descriptive statistics, missing values, negative population entries, distributions, and feature correlations.
4. **Feature preparation:** separate target and identifier, split labeled rows for an initial holdout, and add imputation/scaling in modeling pipelines.
5. **Modeling:** compare a mean baseline, linear models, polynomial regression, regularized models, a decision tree, and a random forest.
6. **Tuning:** inspect decision-tree depth and run a grid search over tree complexity parameters.
7. **Evaluation:** use repeated ShuffleSplit cross-validation and RMSE or MAE, depending on the model experiment.
8. **Prediction:** fit the random forest on all labeled data and write test predictions with `PropertyID` to `submission.csv`.

## Preprocessing

The notebook performs or explores these preparation steps:

- Copies the training and test DataFrames for analysis.
- Uses `TargetPrice` as the label and drops both `TargetPrice` and `PropertyID` from the feature matrix.
- Checks missing values and common string missing markers. In the saved run, `PropertyAge` has **1,313 missing values**; no other missing values were reported in that check.
- Notes 20 rows with negative `NeighborhoodPop` values. The notebook counts these records but does not resolve or exclude them.
- Uses median imputation plus standard scaling in an early preprocessing pipeline; model-specific pipelines later use mean or median imputation, and scaled pipelines for linear models.
- Uses `ShuffleSplit` with 10 splits, each holding out 20% of the labeled rows, for most cross-validation comparisons.

### Preprocessing caveat

The early `transform_pipeline` is fit separately on `X_train` and `X_val`; validation preprocessing should instead call `transform` on a transformer fit to training data, or use a `Pipeline`. The later cross-validation model pipelines correctly place imputation and scaling inside the estimator pipeline. Also, the saved baseline validation code passes raw `X_val` to `DummyRegressor` after training with transformed data; a mean-only regressor ignores feature values, but the inconsistent call should still be cleaned up.

## Exploratory Data Analysis (EDA)

The notebook includes:

- Dataset dimensions, sample records, data types, and descriptive statistics.
- Counts for missing numeric and common string placeholder values.
- Unique-value counts by feature.
- Histograms for numeric fields.
- A Seaborn pair plot, individual feature histograms, and a correlation heatmap/table.
- A check for negative `NeighborhoodPop` entries.

Among the recorded correlations, `TotalRooms` and `TotalBedrooms` have a correlation of approximately **0.79**. This indicates substantial association, but does not by itself mean either feature should be removed. The analysis does not include a formal geographic or time-based validation, and the EDA does not investigate whether records are grouped by neighborhood or time.

## Feature engineering

The notebook's feature matrix includes `RoomsPerHousehold` and `BedroomsRatio`, which are present in the supplied training data. The notebook does not show the formulas used to create these columns, and it does not generate matching fields explicitly for the test data. Before relying on these features, verify that they exist in the test file and document how they were computed. If they are derived from other columns, implement the formulas consistently for both train and test.

`PropertyID` is excluded from training because it is an identifier, then preserved for output. Apart from using the provided ratio features, the notebook does not perform extensive additional feature engineering.

## Choice of machine learning models

The notebook explores models with different complexity and regularization:

- **DummyRegressor:** mean-price baseline.
- **Linear Regression:** interpretable baseline for linear relationships.
- **Polynomial Regression:** adds degree-two terms to model non-linear relationships.
- **Ridge Regression:** linear model with L2 regularization.
- **Lasso Regression:** linear model with L1 regularization.
- **Decision Tree Regressor:** captures non-linear splits; depth is evaluated and tuned.
- **Random Forest Regressor:** ensemble of trees, used for final test predictions.

This progression is useful for comparing simple baselines with more flexible models. Because the best model depends on the evaluation metric and data split, the results should be read with the metric and validation method beside each score.

## Model training and hyperparameter tuning

Most model comparisons use a 10-split `ShuffleSplit` cross-validation scheme with 20% of the data held out in each split. Pipelines fit imputers and, when applicable, scalers within each cross-validation training fold.

The notebook evaluates decision-tree depths from 2 through 20 and unbounded depth, then runs `GridSearchCV` over `max_depth`, `min_samples_leaf`, and `min_samples_split`. The saved best configuration is:

- `max_depth=12`
- `min_samples_leaf=20`
- `min_samples_split=2`
- Best cross-validation RMSE: **0.6224**

The best score is a model-selection estimate, not an independent final test score. The grid search uses all labeled rows (`X`, `y`) after an earlier 80/20 split was created, so that initial split is not a final untouched evaluation set for the tuned tree.

## Performance metrics and recorded results

The notebook uses **RMSE** (root mean squared error) for most regression experiments and **MAE** (mean absolute error) for some linear and polynomial experiments. Lower values indicate smaller errors, but RMSE and MAE are different metrics and should not be compared directly. The target's unit or scale is not documented, so the scores should be interpreted on the dataset's `TargetPrice` scale.

| Experiment | Validation method / metric | Recorded result |
|---|---|---:|
| Mean `DummyRegressor` | Single 80/20 split, RMSE | 1.175 validation RMSE |
| Linear Regression | 10-split CV, RMSE | 0.681 ± 0.025 validation RMSE |
| Degree-2 Polynomial Regression | 10-split CV, MAE | 0.563 ± 0.223 validation MAE |
| Ridge (`alpha=0.5`) | 10-split CV, RMSE | 0.681 ± 0.025 validation RMSE |
| Lasso (`alpha=0.01`) | 10-split CV, MAE | 0.492 ± 0.025 validation MAE |
| Decision Tree (`max_depth=5`) | 10-split CV, RMSE | 0.746 ± 0.014 validation RMSE |
| Tuned Decision Tree | Grid search, CV RMSE | 0.6224 best CV RMSE |
| Random Forest (`n_estimators=300`) | 10-split CV, RMSE | 0.517 ± 0.011 validation RMSE |

The Random Forest has the lowest reported RMSE among the listed RMSE experiments. Its training RMSE is 0.191, while its validation RMSE is 0.517, suggesting a generalization gap and potential overfitting. The polynomial model's MAE varies substantially across splits (standard deviation 0.223), so its mean should be treated cautiously.



