Ensemble of ensembles - model stacking
Ensemble with different types of classifiers:
Different types of classifiers (E.g., logistic regression, decision trees, random forest, etc.) are fitted on the same training data
Results are combined based on either
majority voting (classification) or
average (regression)
Ensemble with a single type of classifier:
Bootstrap samples are drawn from training data
With each bootstrap sample, model (E.g., Individual model may be decision trees, random forest, etc.) will be fitted
All the results are combined to create an ensemble.
Suitabe for highly flexible models that is prone to overfitting / high variance.
Combining Method
Majority voting or average:
Classification: Largest number of votes (mode)
Regression problems: Average (mean).
Method of application of meta-classifiers on outcomes:
Binary outcomes: 0 / 1 from individual classifiers
Meta-classifier is applied on top of the individual classifiers.
Method of application of meta-classifiers on probabilities:
Probabilities are obtained from individual classifiers.
Applying meta-classifier



