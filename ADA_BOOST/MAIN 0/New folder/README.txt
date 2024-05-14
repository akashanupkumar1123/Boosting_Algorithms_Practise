The basic intuition behind Adaboost Classifier
So what’s the basic intuition behind the AdaBoost classifier? We can understand this by discussing the below three points.

Weak Learners: Unlike random forests, which rely on creating learners that are low bias and high variance, In Adaboost, we make learners with high bias and low variance, and these learners are called decision stumps with a depth of only one. Since it has a depth of only one, it uses a single feature to classify the output, which can lead to multiple errors, and thus, Adaboost combines these weak learners to classify the final output.


Weights: In the case of random forest, we would consider the majority value of all the decision trees during the final output. But, in the case of Adaboost, each learner is assigned a weight that indicates the trust in the learner. The higher the weight, the more it contributes to the final output.
f = α1H1 + α2H2 + α3H3 + …




where α indicates the weights and H represents the weak learners

Dependency: Unlike random forests where each decision tree is independent of others, the decision stumps in AdaBoost depend on the previous stumps to make it better at each stage by transmitting the errors of the previous stump. This enables the classifier to progress towards lower bias.




Fit:
Step-1 : 
Initialize weights. wi = C , i = 1,2,..N
This constant can be anything. I will be using 1/N as my constant. Any constant you pick will give exact same performance given it doesn’t cause overflow.
Step-2: 
For m = 1 to M:    
a) Fit classifier Hm with weights w    
b) Compute errₘ = SUM(wi*I(yi!=Hm(xi) ) / SUM(wi)    
c) Compute αₘ = log( (1-errₘ)/errₘ )    
d) Update the weights wi = wi * exp(αₘ*I(yi!=Hm(xi))
Predict:
f(x) = sign( SUM (αₘ*Hm(x)) )


Since we are implementing from scratch, all we need is NumPy. I have imported few sklearn datasets to test our algorithm.



Next, we will implement the decision stump as a class. Remember decision stump is a tree with depth 1.


Here polarity parameter controls whether we want our ‘Yes’ classifications to be 1 or -1. The threshold is set to decide the split and the feature_idx is used to make the stump on that feature. Based on polarity, we make the preds array to contain +1 and -1, which is decided using the threshold.

This stump serves as weak learners for our AdaBoost classifier.


In the fit method, we first create a weight vector that initializes all the weights to 1/n_samples. This is achieved with the help of np.full() method to fill in equal weights.


w = np.full(n_samples, (1/n_samples))


We find the best decision stump based on minimum error for each n_clf (i.e. the number of weak learners). We initialize this min_error to be infinite at first and keep lowering it for every feature and find the lowest one.


n brief, for each feature, we loop over the unique thresholds that the feature offers, predict based on that threshold, and calculate the error which is just the sum of all weights of misclassified samples.

Remember, if the error is greater than 0.5, we flip both the polarity and the error. We then compare this error with our min_error and update accordingly.



Next, we calculate the alpha for the weak learner, which will indicate its performance and help update the weights.


We use a small constant here, EPS, to avoid any undefined cases in the logarithm.


In the predict method, as discussed above, we multiply alpha with the weak learners for all stumps, and the final sign output which we get is the final prediction. And that’s it; we are done.

