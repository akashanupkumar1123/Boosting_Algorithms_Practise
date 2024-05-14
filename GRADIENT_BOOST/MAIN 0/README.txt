Gradient boosting is one of the ensemble machine learning techniques. It uses weak learners like the others in a sequence to produce a robust model.

It is a flexible and powerful technique that can be used for both regression and classification problems. Good results can be achieved even with a very little tuning. It can handle a large number of features and is not biased towards any particular feature type.

On the other hand, it is more sensitive to overfitting than other machine learning methods and can be slow to train, especially on large datasets.

Despite its disadvantages, gradient boosting is a popular method for many machine learning tasks, due to its flexibility, power, and relatively good performance.

In this blog post, I will examine the application of the gradient boosting technique to regression problems. In another article, I will deal with the issue of classification problems.



As you may recall, AdaBoost used decision trees with a depth of 1 called a stump. Each new stump decreases or increases the weights of observations according to the previous stump’s error.

Gradient Boost, on the other hand, starts with a single leaf first, an initial guess. Later, it builds trees. However, unlike AdaBoost, these trees are usually larger than a stump. People usually use decision trees with 8 to 32 leaves in this technique. Again, unlike AdaBoost, the Gradient Boosting technique scales trees at the same rate.

Pseudocode
To begin with, we have a dataset of x, observations , and y, target features. In addition, we have a differentiable loss function. We use a loss function to evaluate our estimations.





