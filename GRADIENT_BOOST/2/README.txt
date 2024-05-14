Gradient Boosting is a machine learning algorithm, used for both classification and regression problems. It works on the principle that many weak learners (eg: shallow trees) can together make a more accurate predictor.



How does Gradient Boosting Works?
Gradient boosting works by building simpler (weak) prediction models sequentially where each model tries to predict the error left over by the previous model.

But, what is a weak learning model?

A model that does slightly better than random predictions is a weak learner. I will show you the exact formula shortly. But for clearly understanding the underlying principles and working of GBT, it’s important to first learn the basic concept of ensemble learning.

This tutorial will take you through the concepts behind gradient boosting and also through two practical implementations of the algorithm:

Gradient Boosting from scratch
Using the scikit-learn in-built function.





Ensemble Learning
Why boosting?
AdaBoost
Gradient Boost
Gradient Boosting Decision Trees
Learning rate and n_estimators (hyperparameters)
Gradient Boosting Algorithm
Algorithm
Implementation
Implementation from scratch
Implementation using scikit-learn
Improving model perfomance
Stochastic Gradient Boosting
Shrinkage
Regularization
Tree Constraints



Ensemble Learning
Ensemble learning, in general, is a model that makes predictions based on a number of different models. By combining a number of different models, an ensemble learning tends to be more flexible (less bias) and less data sensitive (less variance). The two most popular ensemble learning methods are bagging and boosting.

Bagging : Training a bunch of models in parallel way. Each model learns from a random subset of the data, where the dataset is same size as original but is randomly sampled with replacement (bootstrapped).
Boosting : Training a bunch of models sequentially. Each model learns from the mistakes of the previous model. That is, the subsequent models tries to explain and predict the error left over by the previous model.



The application of bagging is found in Random Forests. Random forests are a parallel combination of decision trees. Each tree is trained on random subset of the same data and the results from all trees are averaged to find the classification. The application of boosting is found in Gradient Boosting Decision Trees, about which we are going to discuss in more detail.

Why Boosting?

Boosting works on the principle of improving mistakes of the previous learner through the next learner.

In boosting, weak learners (ex: decision trees with only the stump) are used which perform only slightly better than a random chance. Boosting focuses on sequentially adding up these weak learners and filtering out the observations that a learner gets correct at every step.

Basically, the stress is on developing new weak learners to handle the remaining difficult observations at each step. One of the very first boosting algorithms developed was Adaboost.

Gradient boosting improvised upon some of the features of Adaboost to create a stronger and more efficient algorithm. Let’s look at a brief overview of Adaboost.

AdaBoost
Adaboost uses decision stumps as weak learners.

Decision stumps are nothing but decision trees with only one single split. It also attached weights to observations, adding more weight to ‘difficult-to-classify’ observations and less weight to those that are easy to classify.

The aim is to put stress on the difficult to classify instances for every new weak learner. So, for the next subsequent model, the misclassified observations will receive more weight, as a result, in the new dataset these observations are sampled more number of times according to their new weights, giving the model a chance to learn more of such records and classify them correctly.

As a result, misclassifying the ‘difficult-to-classify’ would be discouraged. Gradient boosting algorithm is slightly different from Adaboost. How? Gradient boosting simply tries to explain (predict) the error left over by the previous model.

And since the loss function optimization is done using gradient descent, and hence the name gradient boosting. Further, gradient boosting uses short, less-complex decision trees instead of decision stumps.

To understand this in more detail, let’s see how exactly a new weak learner in gradient boosting algorithm learns from the mistakes of previous weak learners.

Gradient Boosting Decision Trees
In gradient boosting decision trees, we combine many weak learners to come up with one strong learner.

The weak learners here are the individual decision trees. All the trees are connected in series and each tree tries to minimize the error of the previous tree. Due to this sequential connection, boosting algorithms are usually slow to learn (controllable by the developer using the learning rate parameter), but also highly accurate.

In statistical learning, models that learn slowly perform better. The weak learners are fit in such a way that each new learner fits into the residuals of the previous step so as the model improves. The final model adds up the result of each step and thus a stronger learner is eventually achieved.




A loss function is used to detect the residuals.

For instance, mean squared error (MSE) can be used for a regression task and logarithmic loss (log loss) can be used for classification tasks. It is worth noting that existing trees in the model do not change when a new tree is added.

The added decision tree fits the residuals from the current model.

Understanding the Hyperparameters: Learning rate and n_estimators
Hyperparameters are key parts of learning algorithms which effect the performance and accuracy of a model.

The Learning rate and n_estimators are two critical hyperparameters for gradient boosting decision trees. Learning rate, denoted as α, controls how fast the model learns.

This is done by multiplying the error in previous model with the learning rate and then use that in the subsequent trees.

So, the lower the learning rate, the slower the model learns. Each tree added modifies the overall model.

The advantage of slower learning rate is that the model becomes more robust and efficient and avoids overfitting. In statistical learning, models that learn slowly perform better.

However, learning slowly comes at a cost. It takes more time to train the model which brings us to the other significant hyperparameter. n_estimator is the number of trees used in the model. If the learning rate is low, we need more trees to train the model. However, we need to be very careful at selecting the number of trees. It creates a high risk of overfitting to use too many trees.

Note
One problem that we may encounter in gradient boosting decision trees but not random forests is overfitting due to the addition of too many trees. In random forests, the addition of too many trees won’t cause overfitting.

The accuracy of the model doesn’t improve after a certain point but no problem of overfitting is faced. On the other hand, in gradient boosting decision trees we have to be careful about the number of trees we select, because having too many weak learners in the model may lead to overfitting of data.

Therefore, gradient boosting decision trees require very careful tuning of the hyperparameters.

Gradient Boosting Algorithm
Till now, we have seen how gradient boosting works in theory.

Now, we will dive into the maths and logic behind it so that everything is very clear. Let’s discuss the algorithm step-by-step and make a python program that applies this algorithm to real-time data.

First, let’s go over the basic principle behind gradient boosting once again. Your main aim is to predict a y given a set of x.

The difference between the prediction and the actual value is known as the residual (or in this case, pseudo residuals), on the basis of which the gradient boosting builds successive trees. We know this.

The Algorithm
Let’s say the output model y when fit to only 1 decision tree, is given by:
y=A1+(B1∗X)+e1
where, e_1 is the residual from this decision tree.

In gradient boosting, we fit the consecutive decision trees on the residual from the last one. So when gradient boosting is applied to this model, the consecutive decision trees will be mathematically represented as:
e1=A2+(B2∗X)+e2
and
e2=A3+(B3∗X)+e3

Note that here we stop at 3 decision trees, but in an actual gradient boosting model, the number of learners or decision trees is much more. Combining all three equations, the final model of the decision tree will be given by:
y=A1+A2+A3+(B1∗x)+(B2∗x)+(B3∗x)+e3










1)Calculate error residuals. Actual target value, minus predicted target value [e1= y – y_predicted1 ]
2)Fit a new model on error residuals as target variable with same input variables [call it e1_predicted]
3)Add the predicted residuals to the previous predictions [y_predicted2  = y_predicted1 + e1_predicted]
4)Fit another model on residuals that is still left. i.e. [e2 = y – y_predicted2] and repeat steps 2 to 5 until it starts overfitting or the sum of residuals become constant. Overfitting can be controlled by consistently checking accuracy on validation data.


The code above is a very basic implementation of gradient boosting trees. The actual libraries have a lot of hyperparameters that can be tuned for better results. This can be better understood by using the gradient boosting algorithm on a real dataset.

Implementation using scikit-learn
For implementation on a dataset, we will be using the PIMA Indians Diabetes dataset, which has information about a an individual’s health parameters and an output of 0 or 1, depending on whether or not he has diabetes.

The task here is classify a individual as diabetic, when given the required inputs about his health. First, let’s import all required libraries.





The model has been trained and we can now observe the outputs as well. Below, you can see the confusion matrix of the model, which gives a report of the number of classifications and misclassifications.

The number of misclassifications by the Gradient Boosting Classifier are 42, comapared to 112 correct classifications. The model has performed decently. Let’s check the accuracy





This can be improved by tuning the hyperparameters or processing the data to remove outliers.| The discussion above is just the tip of iceberg when it comes to gradient boosting. The underlying concepts can be understood in more detail by starting with the very basics of machine learning algorithms and understanding the working of python code. This however gives us the basic idea behind gradient boosting and its underlying working principles.

Improving performance of gradient boosted decision trees
As we already discussed above, gradient boosting algorithms are prone to overfitting and consequently poor performance on test dataset. There are some pointers you can keep in mind to improve the performance of gradient boosting algorithm.

Stochastic Gradient Boosting
Stochastic gradient boosting involves subsampling the training dataset and training individual learners on random samples created by this subsampling. This reduces the correlation between results from individual learners and combining results with low correlation provides us with a better overall result. A few variants of stochastic boosting that can be used:

Subsample rows before creating each tree.
Subsample columns before creating each tree.
Subsample columns before considering each split.

Shrinkage
The predictions of each tree are added together sequentially. Instead, the contribution of each tree to this sum can be weighted to slow down the learning by the algorithm.

This weighting is called a shrinkage or a learning rate. Using a low learning rate can dramatically improve the performance of your gradient boosting model.

Usually a learning rate in the range of 0.1 to 0.3 gives the best results. Keep in mind that a low learning rate can significantly drive up the training time, as your model will require more number of iterations to converge to a final loss value.

Regularization
L1 and L2 regularization penalties can be implemented on leaf weight values to slow down learning and prevent overfitting. Gradient tree boosting implementations often also use regularization by limiting the minimum number of observations in trees’ terminal nodes

Tree Constraints
There are a number of ways in which a tree can be constrained to improve performance.

Number of trees : Adding excessive number of trees can lead to overfitting, so it is important to stop at the point where the loss value converges.
Tree depth : Shorter trees are preferred over more complex trees. Limit the number of levels in a tree to 4-8.
Minimum improvement in loss : If you have gone over the process of decision making in decision trees, you will know that there is a loss associated at each level of a decision tree. A minimum improvement in loss required to build a new level in a tree can be predecided. This helps in pruning the tree to keep it short.
Number of observations per split :This imposes a minimum constraint on the amount of training data at a training node before a split can be considered







