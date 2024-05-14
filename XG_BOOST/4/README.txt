XGBoost (Extreme Gradient Boosting)
Documentation

tqchen github

dmlc github

“Gradient Boosting” is proposed in the paper Greedy Function Approximation: A Gradient Boosting Machine, by Friedman.

XGBoost is based on this original model.

Supervised Learning

Objective Function : Training Loss + Regularization
𝑂𝑏𝑗(Θ)=𝐿(θ)+Ω(Θ)
 
𝐿  is the training loss function, and
Ω  is the regularization term.
Training Loss
The training loss measures how predictive our model is on training data.

Example 1, Mean Squared Error for Linear Regression:

𝐿(θ)=∑𝑖(𝑦𝑖−𝑦̂ 𝑖)2
 
Example 2, Logistic Loss for Logistic Regression:

𝐿(θ)=∑𝑖[𝑦𝑖𝑙𝑛(1+𝑒−𝑦̂ 𝑖)+(1−𝑦𝑖)𝑙𝑛(1+𝑒𝑦̂ 𝑖)]
 
Regularization Term
The regularization term controls the complexity of the model, which helps us to avoid overfitting.


Specifically, xgboost used a more regularized model formalization to control over-fitting, which gives it better performance.

For model, it might be more suitable to be called as regularized gradient boosting.


