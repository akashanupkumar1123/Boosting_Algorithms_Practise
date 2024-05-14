AdaBoost / Adaptive Boosting
Robert Schapire

Wikipedia

Chris McCormick

Scikit Learn AdaBoost

1995

As above for Boosting:

Similar to human learning, the algo learns from past mistakes by focusing more on difficult problems it did not get right in prior learning.
In machine learning speak, it pays more attention to training instances that previously underfitted.
Source: Scikit-Learn:

Fit a sequence of weak learners (i.e., models that are only slightly better than random guessing, such as small decision trees) on repeatedly modified versions of the data.
The predictions from all of them are then combined through a weighted majority vote (or sum) to produce the final prediction.
The data modifications at each so-called boosting iteration consist of applying weights  𝑤1,𝑤2,…,𝑤𝑁  to each of the training samples.
Initially, those weights are all set to  𝑤𝑖=1/𝑁 , so that the first step simply trains a weak learner on the original data.
For each successive iteration, the sample weights are individually modified and the learning algorithm is reapplied to the reweighted data.
At a given step, those training examples that were incorrectly predicted by the boosted model induced at the previous step have their weights increased, whereas the weights are decreased for those that were predicted correctly.
As iterations proceed, examples that are difficult to predict receive ever-increasing influence. Each subsequent weak learner is thereby forced to concentrate on the examples that are missed by the previous ones in the sequence.






Gradient Boosting / Gradient Boosting Machine (GBM)
Works for both regression and classification

Wikipedia

Sequentially adding predictors
Each one correcting its predecessor
Fit new predictor to the residual errors
Compare this to AdaBoost:

Alter instance weights at every iteration
*Step 1. *

𝑌=𝐹(𝑥)+𝜖
*Step 2. *

𝜖=𝐺(𝑥)+𝜖2
Substituting (2) into (1), we get:

𝑌=𝐹(𝑥)+𝐺(𝑥)+𝜖2
*Step 3. *

𝜖2=𝐻(𝑥)+𝜖3
Now:

𝑌=𝐹(𝑥)+𝐺(𝑥)+𝐻(𝑥)+𝜖3
Finally, by adding weighting

𝑌=𝛼𝐹(𝑥)+𝛽𝐺(𝑥)+𝛾𝐻(𝑥)+𝜖4
Gradient boosting involves three elements:

Loss function to be optimized: Loss function depends on the type of problem being solved. In the case of regression problems, mean squared error is used, and in classification problems, logarithmic loss will be used. In boosting, at each stage, unexplained loss from prior iterations will be optimized rather than starting from scratch.

Weak learner to make predictions: Decision trees are used as a weak learner in gradient boosting.

Additive model to add weak learners to minimize the loss function: Trees are added one at a time and existing trees in the model are not changed. The gradient descent procedure is used to minimize the loss when adding trees.

