[![Build Status](https://travis-ci.org/pplonski/gafe.svg?branch=master)](https://travis-ci.org/pplonski/gafe)
[![Coverage Status](https://coveralls.io/repos/github/pplonski/gafe/badge.svg?branch=master)](https://coveralls.io/github/pplonski/gafe?branch=master)

# GAFE - Genetic Algorithm Feature Engineering

Simple algorithm for new features engineering.

 - gafe tries different combination of features with operators: `+`, `-`, `*`
 - add it to your dataset
 - re-evaluate the classifier performance with new features

## Example

1. In binary classification problem, you have dataset with following 20 features: `feature1`, `feature2`, `feature3`, ..., `feature20` and binary target column.
2. GAFE computes the base score for your dataset using Random Forest (32 trees), 5-fold CV and negative log loss.
3. The algorithm is starting with random population of new feature sets. Each new feature set contains from `new_features_lower_cnt` to `new_features_upper_cnt` new features. Each new feature is combination of original features with operators: `+`, `-`, `*`, for example new feature can look like: `feature1-feature2-feature3`.
4. Each new feature set is scored with the same classifier as in step 2. For scoring are used concatenated original and new features.
5. The genetic algorithm is applied to mutate new feature sets to find better features.
6. At the end, the best feature set is selected based on classifier performance.
