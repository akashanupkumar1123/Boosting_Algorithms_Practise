we're going to implement the core statistical learning algorithm of XGBoost, including most of the key hyperparameters and their functionality. Our implementation will also support user-defined custom objective functions, meaning that it can perform regression, classification, and whatever exotic learning tasks you can dream up, as long as you can write down a twice-differentiable objective function. We'll refrain from implementing some simple features like column subsampling which will be left to you, gentle reader, as exercises. In terms of tree methods, we're going to implement the exact tree-splitting algorithm, leaving the sparsity-aware method (used to handle missing feature values) and the approximate method (used for scalability) as exercises or maybe topics for future posts.

As always, if something is unclear, try backtracking through the previous posts on gradient boosting and decision trees to clarify your intuition. We've already built up all the statistical and computational background needed to make sense of this scratch build. Here are the most important prerequisite posts:

Gradient Boosting Machine from Scratch
Decision Tree From Scratch
How to Understand XGBoost
Great, let's do this.

The XGBoost Model Class
We begin with the user-facing API for our model, a class called XGBoostModel which will implement gradient boosting and prediction. To be more consistent with the XGBoost library, we'll pass hyperparameters to our model in a parameter dictionary, so our init method is going to pull relevant parameters out of the dictionary and set them as object attributes. Note the use of python's defaultdict so we don't have to worry about handling key errors if we try to access a parameter that the user didn't set in the dictionary.





The fit method, based on our classic GBM, takes a feature dataframe, a target vector, the objective function, and the number of boosting rounds as arguments. The user-supplied objective function should be an object with loss, gradient, and hessian methods, each of which takes a target vector and a prediction vector as input; the loss method should return a scalar loss score, the gradient method should return a vector of gradients, and the hessian method should return a vector of hessians.

In contrast to boosting in the classic GBM, instead of computing residuals between the current predictions and the target, we compute gradients and hessians of the loss function with respect to the current predictions, and instead of predicting residuals with a decision tree, we fit a special XGBoost tree booster (which we'll implement in a moment) using the gradients and hessians. I've also added row subsampling by drawing a random subset of instance indices and passing them to the tree booster during each boosting round. The rest of the fit method is the same as the classic GBM, and the predict method is identical too.




The XGBoost Tree Booster
The XGBoost tree booster is a modified version of the decision tree that we built in the decision tree from scratch post. Like the decision tree, we recursively build a binary tree structure by finding the best split rule for each node in the tree. The main difference is the criterion for evaluating splits and the way that we define a leaf's predicted value. Instead of being functions of the target values of the instances in each node, the criterion and predicted values are functions of the instance gradients and hessians. Thus we need only make a couple of modifications to our previous decision tree implementation to create the XGBoost tree booster.

Initialization and Inserting Child Nodes
Most of the init method is just parsing the parameter dictionary to assign parameters as object attributes. The one notable difference from our decision tree is in the way we define the node's predicted value. We define self.value according to equation 5 of the XGBoost paper, a simple function of the gradient and hessian values of the instances in the current node. Of course the init also goes on to build the tree via the maybe insert child nodes method. This method is nearly identical to the one we implemented for our decision tree. So far so good.


Split Finding
Split finding follows the exact same pattern that we used in the decision tree, except we keep track of gradient and hessian stats instead of target value stats, and of course we use the XGBoost gain criterion (equation 7 from the paper) for evaluating splits.


Prediction
Prediction works exactly the same as in our decision tree, and the methods are nearly identical.

We use the scikit learn California housing dataset for benchmarking.



Let's start with a nice friendly squared error objective function for training. We should probably have a future post all about how to define custom objective functions in XGBoost,




Here I use a more or less arbitrary set of hyperparameters for training. Feel free to play around with tuning and trying other parameter combinations yourself.




Wrapping Up
I'd say this is a pretty good milestone for us here at Random Realizations. We've been hammering away at the various concepts around gradient boosting, leaving a trail of equations and scratch-built algos in our wake. Today we put all of that together to create a legit scratch build of XGBoost,