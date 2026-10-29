# System Threat Classification

A machine learning project that explores system telemetry and predicts whether a device is in the positive or negative threat class. The notebook follows the workflow from problem framing and data inspection through feature preparation, model comparison, hyperparameter search, and creation of predictions for the unlabeled test set.

> **Project status:** This is an educational modeling project based on a competition dataset. The reported scores are validation results from one stratified holdout split, not evidence of production performance or realized business impact.

## Contents

## Business problem

Organizations manage fleets of computers with different operating systems, hardware configurations, security settings, and protection states. A system that is more likely to be threatened may warrant closer review or additional protective action. Reviewing every device manually is costly, so a classifier could help prioritize devices for investigation.

This project frames that as a **binary classification** task: use a device's recorded system and security attributes to predict its `target` label. The model is a decision-support experiment; it does not diagnose malware, establish causality, or take actions on devices. The original data documentation should be consulted for the precise target-label definition and collection process.

### Intended business value

If a model were validated and deployed responsibly, it could help an operations or security team rank devices for follow-up, allocate analyst time, and identify patterns associated with higher threat labels. This project does not measure saved time, prevented incidents, financial benefit, or operational outcomes. Those impacts would need to be evaluated in a pilot against an existing workflow.

## Dataset selection

The notebook uses the **System-Threat-Forecaster** dataset supplied as separate labeled training and unlabeled test files. The features describe device, operating system, hardware, and security configuration. The labeled file contains the `target` column; the test file is used to demonstrate generating predictions.

The dataset is suitable for a tabular classification exercise because it contains mixed categorical and numerical fields, missing values, high-cardinality fields, and a binary target. It also provides a separate test file for a final prediction workflow.

### Obtain and place the data

The CSVs are not redistributed here. Obtain them from the original competition or data source, and arrange the project like this:

```text
project/
├── Classification.ipynb
├── README.md
├── requirements.txt
└── data/
    ├── train.csv
    └── test.csv
```

The notebook and script expect the paths `data/train.csv` and `data/test.csv`, relative to their project directory. The labeled training file must include `target`. Do not commit data if its source terms prohibit redistribution; check the applicable license and competition rules first.

## Machine learning lifecycle

This project demonstrates the following lifecycle:

1. **Define the problem:** classify devices into the dataset's binary target classes, with the potential use case of prioritizing review.
2. **Understand the data:** inspect shapes, samples, data types, distributions, missingness, unique values, and relationships to the target.
3. **Prepare the data:** identify constant and ID-like fields, resolve feature types, engineer a small number of fields, split labeled data, and construct preprocessing pipelines.
4. **Train candidate models:** fit linear, tree-based, and boosting classifiers using a consistent transformed feature matrix.
5. **Tune selected models:** use randomized and grid search with cross-validation on the training partition.
6. **Evaluate:** compare holdout accuracy and, for some models, class-level precision, recall, and F1.
7. **Generate predictions:** transform the provided unlabeled test set and produce a two-column prediction file in the notebook.
8. **Plan deployment and monitoring:** not implemented here; any operational use would require additional validation, controls, and monitoring.

## Data preprocessing

The notebook's preprocessing and preparation work includes:

- Copies the original training and test frames for analysis.
- Removes `MachineID` from the analysis data as an identifier-like field.
- Counts nulls and several string placeholders (`NA`, `N/A`, `-`, `?`, `UNKNOWN`, and `Unknown`) and inspects unique-value counts. These checks help reveal data-quality patterns; they do not standardize every placeholder across all columns.
- Separates object columns from numeric columns and reviews missingness and cardinality in each group.
- Uses a variance threshold to identify zero-variance numeric fields.
- Treats selected numeric-coded fields (including region, OS suite/build, and protection-state fields) as categorical. Several ID-like numeric fields are also identified for categorical treatment.
- Converts selected categorical columns to object type and excludes selected identifier or near-identifier fields from the chosen feature lists.
- Splits the labeled data into training and validation partitions with a 70/30 split, stratified by `target`, using `random_state=42`. The saved run reports 70,000 training rows and 30,000 validation rows.
- Uses a `ColumnTransformer` and pipelines to impute missing values, encode categorical values, and scale continuous numerical fields.

For categorical data, the notebook uses most-frequent imputation. Lower-cardinality categories are one-hot encoded; higher-cardinality categories are target encoded. For numeric fields, it uses most-frequent imputation for selected binary features and median imputation plus standard scaling for selected continuous features. The split occurs before fitting the preprocessing transformer, helping avoid fitting imputers/scalers on the validation rows.

## Exploratory data analysis

The notebook explores the data before modeling through:

- Dataset shape, sample rows, data types, descriptive statistics, and unique-value counts.
- Missing-value counts and categorical placeholder checks. The notebook notes that 33 features have null values in the inspected training data.
- Categorical value-count summaries, missingness, unknown values, frequency plots, and target proportions.
- Train/test category and numeric distribution comparisons. These are visual checks rather than formal drift tests.
- Chi-square tests to explore associations between categorical fields and the target. Statistical significance is treated as exploratory evidence, not proof that a feature is useful or causal.
- Numeric histograms, violin plots, box plots, variance checks, and a correlation heatmap.

The EDA suggests that some numeric-coded fields are categorical, some features are highly skewed, there are outliers, and some numeric fields are correlated. The notebook keeps outliers rather than removing them because similar values appear in the test data. This choice was exploratory and was not verified through a dedicated robustness experiment.

## Feature engineering

The notebook performs or considers these feature changes:

- Drops `MachineID` from the modeled feature data because it behaves like an identifier.
- Identifies constant numeric columns and excludes them from the numerical feature set.
- Reclassifies selected integer-coded fields as categorical so that they can be encoded rather than treated as continuous measurements.
- Creates a display-density proxy called `PPI` from horizontal resolution, vertical resolution, and display diagonal. The formula in the notebook is a custom proxy and should be reviewed before describing it as standard pixels-per-inch.
- Converts memory and storage capacities from MB to GB and renames the affected columns.
- Replaces very large battery charge counts (over 100,000) with missing values as an outlier handling rule.
- Groups categorical features by cardinality to choose between one-hot encoding and target encoding.

These transformations reflect exploratory choices. A future iteration should validate each transformation with clear domain reasoning and compare performance with and without it.

## Models

The notebook compares several model families:

- **Logistic Regression:** a simple linear baseline that is relatively interpretable.
- **Decision Tree:** a non-linear model that can capture feature interactions, but can overfit without constraints.
- **Random Forest:** an ensemble of trees intended to improve stability and capture non-linear patterns.
- **XGBoost:** gradient-boosted trees.
- **LightGBM:** gradient-boosted trees designed for efficient training on tabular data.
- **Stacking Classifier:** combines base estimators with a logistic regression meta-learner.

The selection spans a baseline linear model and tree ensemble approaches, which is a reasonable first comparison for mixed, tabular data. The notebook's final test-set prediction cell uses a fitted Random Forest model.

## Training, evaluation, and metrics

Candidate models are trained on the 70% training partition. The 30% stratified validation partition is used to estimate performance. Accuracy is reported for most models; the untuned Logistic Regression also reports a classification report with per-class precision, recall, and F1-score.

### Saved holdout results

The following are the validation accuracies recorded in the original notebook. Scores are rounded to four decimal places.

| Model / run | Validation accuracy | Notes |
|---|---:|---|
| Logistic Regression, initial | 0.5720 | Classification report included in notebook. |
| Logistic Regression, tuned | 0.5725 | Randomized search selected `C=0.5908`, `max_iter=1000`, `solver=liblinear`; a subsequent grid search recorded 0.5724. |
| Decision Tree, initial | 0.5407 | Training accuracy was 1.0000, indicating substantial overfitting. |
| Decision Tree, tuned | 0.5578 | Grid search selected entropy criterion, max depth 8, minimum leaf size 14, and minimum split size 2. |
| Random Forest, initial | 0.5803 | Training accuracy was 1.0000, indicating substantial overfitting. |
| Random Forest, tuned | 0.5814 | Randomized search selected 400 trees, max depth 20, minimum leaf size 1, minimum split size 2. |
| XGBoost, initial | 0.5750 | — |
| XGBoost, tuned | 0.5771 | Best cross-validation score was also recorded; see tuning note below. |
| LightGBM, initial | 0.5765 | — |
| Stacking Classifier | 0.5765 | — |

The saved run's Logistic Regression classification report shows precision, recall, and F1 near 0.56–0.58 for both classes, with overall accuracy 0.572. Class supports are 14,843 for class 0 and 15,157 for class 1 in that validation split.

**Interpretation:** tuned Random Forest had the highest listed holdout accuracy at approximately 58.14%, a modest improvement over the untuned baseline. This is not enough evidence by itself to select a deployment model. The class distribution is relatively balanced in the recorded validation set, but accuracy alone does not express the cost of false negatives versus false positives.

### Hyperparameter tuning

The notebook uses `RandomizedSearchCV` and `GridSearchCV` with accuracy as the search score. Logistic Regression and Decision Tree receive randomized and grid searches; Random Forest and boosting models receive randomized searches. Search uses three- or five-fold cross-validation depending on the model.

The tuned XGBoost cell records a best cross-validation score of approximately 0.7800 but a holdout accuracy of 0.5771. These are different evaluation quantities and should not be compared as though they were the same score. The gap is unusually large and warrants checking the notebook's target encoding, preprocessing, parameter search and evaluation flow before relying on that tuning result.

## Challenges and limitations

- **Missing values and placeholders:** missingness appears in many features, while string placeholders can represent unknown or unavailable values. The current checks do not fully unify these representations.
- **Mixed feature types:** some integer columns represent categories or identifiers, so type-based handling alone is insufficient.
- **High cardinality:** one-hot encoding can expand the feature space; target encoding has its own leakage and validation risks if not carefully cross-fitted.
- **Overfitting:** the untuned Decision Tree and Random Forest reach perfect training accuracy but much lower validation accuracy.
- **Metric choice:** accuracy does not encode operational costs or provide a threshold tradeoff. In a security setting, recall for the threat class and precision at an actionable alert volume may matter more.
- **Tuning/evaluation inconsistency:** the saved XGBoost cross-validation score differs sharply from its holdout score and should be investigated.
- **Notebook portability:** the original notebook was created in Kaggle and depends on the external data files and compatible package versions. This cleaned copy uses local relative data paths.
- **Reproducibility:** `requirements.txt` lists package names but does not lock exact versions, and the notebook contains exploratory choices that require review before production use.
- **No deployment evaluation:** inference latency, calibration, threshold selection, monitoring, privacy, security review, and business outcomes are not measured.

## Business impact

No realized business impact can be claimed from this notebook. The project demonstrates how device telemetry could be used to support threat prioritization, but it does not establish that the classifier improves incident detection, reduces analyst workload, or prevents losses. A business impact study would require a defined workflow, a selected operating threshold, measured false-positive and false-negative costs, comparison to the current process, and a monitored pilot.

## Future improvements

1. **Confirm target semantics and data provenance.** Document what each target class means, how labels were created, the time period, and the dataset license.
2. **Establish a strong baseline.** Compare against majority-class and simple rule-based baselines, and report balanced accuracy and class-wise metrics.
3. **Use metrics aligned with operations.** Add confusion matrices, precision/recall, F1, ROC-AUC, PR-AUC, and threshold analysis; prioritize recall or precision based on stakeholder costs.
4. **Audit validation and leakage.** Put all preprocessing and target encoding inside a cross-validated pipeline, use out-of-fold encoding where appropriate, and preserve a final untouched test split. Investigate the XGBoost CV/holdout discrepancy.
5. **Improve the validation design.** Consider repeated stratified cross-validation or a time-based split if records have temporal ordering, and quantify score variability.
6. **Refine data cleaning and features.** Normalize missing-value markers consistently, verify units and custom formulas such as `PPI`, and evaluate feature groups and outlier treatment systematically.
7. **Control overfitting and complexity.** Tune tree regularization and compare simpler models; assess feature importance with permutation or SHAP-style methods while accounting for correlated fields.
8. **Reproducibility.** Pin dependency versions, add a data dictionary, record random seeds and experiment settings, and provide a clean environment/setup workflow.
9. **Operationalize only after validation.** Calibrate probabilities, select thresholds with stakeholders, assess subgroup performance, document privacy/security controls, then monitor drift and outcomes in a limited pilot.

## Setup and run

Requires Python 3.10 or newer and the CSV files described above.

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
jupyter notebook Classification.ipynb
```

To run the revised model workflow instead of the exploratory notebook:

```bash
python classification_model.py
```

The script fits a baseline and tunes a Random Forest using cross-validation with preprocessing inside the pipeline. It reports holdout metrics, refits the selected configuration on all labeled rows, and writes `submission.csv` in the project directory. Verify the expected submission ID and column schema against the original competition/source before submitting.

## License and data use

No project or dataset license is specified here. Add an appropriate license before redistribution, and follow the dataset provider's terms for the data and derived outputs.
