# Credit Default Prediction – Give Me Some Credit

This is a machine learning project based on the Kaggle **Give Me Some Credit** competition.

I am currently a third-year university student and I made this project mainly to get more hands-on experience with classification, data preprocessing, model evaluation and hyperparameter tuning on a real dataset.

The goal is to predict whether a person will experience serious financial distress within the next two years.

## Dataset

The dataset contains around **150,000 borrowers** and information such as:

* credit utilization
* age
* monthly income
* debt ratio
* number of open credit lines
* real estate loans
* number of dependents
* several variables describing past late payments

The target variable is:

`SeriousDlqin2yrs`

where `1` means that the borrower experienced a serious delinquency within two years.

The dataset is quite imbalanced, with only around **6.7% positive cases**.

---

## Exploratory Data Analysis

Before training more complex models, I spent quite a lot of time looking at the individual variables and checking whether their values actually made sense.

This ended up being one of the most important parts of the project.

### Late payment special codes

The three late-payment variables contained values `96` and `98`.

These values clearly did not behave like real counts of late payments. There were only **269 rows** containing them and they strongly distorted the logistic regression coefficients.

Instead of treating them as actual counts, I replaced them with `0` and created an additional binary feature:

`special_late_code`

This alone improved the logistic regression ROC-AUC from roughly **0.71 to 0.81**.

### Revolving credit utilization

`RevolvingUtilizationOfUnsecuredLines` had some extreme values.

Most observations were between 0 and approximately 1, while the maximum value was above **50,000**.

The relationship with default risk was fairly logical up to a utilization of around 2:

| Utilization | Default rate |
| ----------- | -----------: |
| 0–0.25      |         2.1% |
| 0.25–0.50   |         5.3% |
| 0.50–0.75   |        10.1% |
| 0.75–1.00   |        18.2% |
| 1.00–1.50   |        39.7% |
| 1.50–2.00   |        44.5% |

Above this level, the relationship became much less meaningful.

I therefore tested capping the feature at `2`.

This was especially important for logistic regression and increased its ROC-AUC by around **0.035**.

XGBoost was much less sensitive to these extreme values, but the cap still produced a small improvement.

### DebtRatio and MonthlyIncome

`DebtRatio` also contained very large values and behaved strangely when `MonthlyIncome` was missing or equal to zero.

I tested several approaches:

* replacing problematic DebtRatio values
* imputing MonthlyIncome with the median
* using a missing-income indicator
* setting unreliable DebtRatio values to missing

None of these changes improved XGBoost consistently.

Because the exact meaning of some of these extreme DebtRatio observations is unclear, I decided not to aggressively modify this feature.

### Age

There was one observation with `age = 0`.

Since this is obviously not a valid value, I treat it as missing.

Other high age values were rare but still possible, so I did not remove them.

---

# Logistic Regression

I first built a logistic regression model as a simple and interpretable baseline.

This was useful because the coefficients made it easy to see when something in the data was wrong.

For example, before fixing the special delinquency codes and utilization outliers, some coefficients had economically unreasonable signs.

After preprocessing, the strongest effects were much more sensible:

* higher revolving utilization → higher default risk
* more late payments → higher default risk
* higher income → lower default risk
* higher age → lower default risk

### Logistic Regression performance

Initial model:

`ROC-AUC ≈ 0.713`

After fixing the delinquency special codes:

`ROC-AUC ≈ 0.816`

After capping revolving utilization:

`ROC-AUC ≈ 0.851`

5-fold cross-validation:

`Mean ROC-AUC ≈ 0.8514`

I also tested removing weak variables such as `DebtRatio` and `NumberOfDependents`.

The performance barely changed, which showed that most of their useful information was already captured by the other features.

---

# XGBoost

I then moved to XGBoost.

Unlike logistic regression, XGBoost can naturally model nonlinear relationships and interactions between variables. It also handles missing values directly, so much less preprocessing was necessary.

Even the basic untuned model already performed better than logistic regression.

### Initial XGBoost

Validation ROC-AUC:

`0.8574`

However, the model was clearly overfitting:

```text
Train AUC: 0.9136
Test AUC:  0.8574
Gap:       0.0563
```

The next step was therefore mainly focused on improving generalization rather than increasing training performance.

---
### XGBoost preprocessing

I kept the preprocessing for XGBoost relatively simple and made only two main feature changes:

* The special delinquency values `96` and `98` were replaced with `0`, while their occurrence was preserved in a separate `special_late_code` feature.
* `RevolvingUtilizationOfUnsecuredLines` was capped at `2`. XGBoost was much less sensitive to the extreme values than logistic regression, but the cap still improved validation ROC-AUC from approximately **0.8574 to 0.8587**.

Other variables were mostly left in their original form because XGBoost can handle nonlinear relationships and missing values directly. I also treated the single invalid `age = 0` observation as missing and tested several approaches for `MonthlyIncome` and `DebtRatio`, but these did not consistently improve performance.

## Hyperparameter tuning

I first used `RandomizedSearchCV` to search a relatively broad parameter space.

This reduced the train-test gap substantially and increased validation performance.

After that, I used smaller grid searches to better understand the effect of individual parameters.

I tested combinations of:

* `max_depth`
* `min_child_weight`
* `subsample`
* `colsample_bytree`
* `gamma`
* `reg_lambda`
* `reg_alpha`
* `learning_rate`
* `n_estimators`

Heatmaps were useful here because I did not want to select a model based only on a single best score. I also wanted to see whether the surrounding parameter combinations performed similarly for overfitting purposes.

For example, the best tree depth was around **4–5**, while deeper trees started to reduce cross-validation performance.

The learning-rate / number-of-trees grid also showed a clear trade-off: smaller learning rates required more trees.

The final area around the optimum was relatively flat, which made me more confident that the result was not based on one lucky hyperparameter combination.

---

## Final XGBoost model

Current parameters:

```python
XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    random_state=30,

    max_depth=5,
    min_child_weight=15,

    subsample=0.7,
    colsample_bytree=0.7,

    gamma=0.5,
    reg_lambda=5,
    reg_alpha=0,

    learning_rate=0.01,
    n_estimators=1000
)
```

Performance:

```text
5-fold CV ROC-AUC:  0.8663
Validation ROC-AUC: 0.8664
Train ROC-AUC:      0.8766
Train-test gap:     0.0102
```

Compared with the original XGBoost model, the training AUC decreased significantly while the validation AUC improved. This suggests that most of the original overfitting was removed.

---

# Kaggle submission

After finishing the main model, I trained it on the full training dataset and submitted predictions for the official Kaggle test set.

The submission achieved:

```text
Public Score:  0.86189
Private Score: 0.86816
```

The **Private Score of 0.86816** is very close to the historical winning solutions, which were around **0.8695 AUC**.

That leaves a gap of only around **0.0014 AUC** to the top historical score.

I was particularly happy with this result because the model is still a single XGBoost model without large-scale feature engineering or model ensembling.

---

# What I learned

The biggest takeaway from this project was that improving the data was often more useful than immediately trying a more complex model.

For logistic regression especially, a few unusual values were enough to completely distort otherwise useful features.

I also got practical experience with:

* exploratory data analysis
* imbalanced binary classification
* missing values and outliers
* feature engineering
* logistic regression
* XGBoost
* ROC-AUC
* cross-validation
* detecting overfitting
* randomized search
* grid search
* hyperparameter heatmaps
* Kaggle submissions

I also learned that the highest CV score is not automatically the best model. Looking at train-validation gaps and the surrounding hyperparameter region was useful for checking whether an improvement was actually stable.

---

# Next steps

For now I consider the main version of the project finished.

I still want to experiment with ways of getting closer to or possibly matching the top Kaggle scores. The next experiments will mainly focus on:

* additional feature engineering
* aggregated late-payment features
* interactions between credit variables
* larger global hyperparameter searches
* alternative boosting models
* model blending / ensembling

These experiments will be an extension of the original project rather than part of the main baseline analysis.
