# BayesianMLOptimization
Hyperparameter Optimization of a SVM classifier with sigmoid kernel  
the maximization of Accuracy in 10 fold cross validation is considered as object function.  
The dataset to use for the project is named "wine", available on "sklearns": use the "load_wine()" command to load the dataset.  
  
  
The SVM classifier with sigmoid kernel has three hyperparameters:

[] C (for regularization)
[] gamma (for sigmoid kernel)
[] coef0 (for sigmoid kernel)

An overall budget of 150 function evaluations must be considered: 30 for initializations + 120 for the sequential optimization.  
Both Gaussian Process and Random Forests must be considered as probabilistic surrogate models.  
Acquisition functions to be used are: Probability of Improvement, Expected Improvement and Upper Confidence Bound.  
A simple grid search with the same budget must be implemented (5 levels for C, 6 levels for gamma and 5 levels for coef0) to build a "baseline".  
Thus, an overall number of 7 experiments must be performed ( 2 surrogate models x 3 acquisition functions + 1 grid search ).  
Important: initialize the random seed so that every SMBO experiments will start from the same initial points.  