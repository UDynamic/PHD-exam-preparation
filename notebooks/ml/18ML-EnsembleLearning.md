I've just mentioned toppics as in a list, develop on all of them:

Ensemble learning introduction of concept

simple summation using different fusion functions:(sum, weighted sum, median, minimum, maximum, product)

for quick remembering of the boostin and bagging purpose, pay attention to the beginning. bag and var has the same tunality oppenning. so boosting would be for bias. also bagging has "a" in it like parallel, is boosting is sequential.

1. sequential Ensemble methods

   1. boosting.

      1. decreases bias
      2. used by combination of simple models with high biases to lower the bias (combining weaker classifiers to form a powerful model)
      3. Decision stump classifier (It's a line) and algorithm and comparison with random forrest
      4. a line as a weak classifier is applicable too
      5. use the weighted combination of the same classifiers with different parameters
         1. at first weights are the same (or fixed ) but gradually increase buy the error meaning that wrong classified datas will have more weight in the next classifier
      6. give complete definition and algorith and formulation of Adaboost algorithm
         1. classifiers added one by one (sequentially)
         2. iteratively fits a classifier
         3. classifiers add up one by one
         4. wrong classified data will have more weight in the next iteration
         5. devides data to simple and complex and uses the simplest data for the first iteration and decreases probability for the next iteration [! elaborate]
         6. at the end uses weigted vote of max for classification. meaning: calculate the sum weighted vote for classifiers at that point for test data, we choose the clas with max weighted sum. : H_m = \alpha_1 h_1 + \dots + \alpha_m h_m and then \hat{y} = sign(H_m(x))
         7. threshold must be less than 0.5
         8. at first all the data is the same \frac{1}{n} weight
         9. at each iteration until M we find the classifier that minimizes the sum weighted error (less than 0.5)
         10. find \epsilon_m as normalized weighted error
         11. find \alph_m from -epsilon_m [! give complete formulation]
         12. update each datas weight using \alpha_m
         13. create the complete classifier and classify on it's sign [! very vague, elaborate]
         14. give the formulation for loss of the m-th weak learner : \sum w_i I(h^i \neq h_m(x^i)) then the \epsilon_m formulation from the exact formula weighted error of the m-th weak learner
         15. \alpha_m = \ln(\\frac{1-\epsilon_m}{\epsilon_m})
         16. loss(y, \hat{y})=e^{y*H_m(x)} equals e for different sign and 1 over e for the same sign
         17. mention and elaborate and deliver analysis on the loss funciton graph
         18. give complete analysis on the relation betwean \epsilon_m and \alpha_m at each iteration (the more the error the less the alpha)
         19. prove with the formulation for the w_{m+1} using w_{m} that noise data won't get much weight
         20. for each classifier if the error is less than 0.5 it's good enough.
         21. at each iteration, if the weighted loss is better than random the exponential loss will definetly decrease. : E_train (H_m(x)) \leq \prod 2\sqrt(\\epsilon_m(1-\epsilon_m))
         22. total train error is decreasing
         23. weighted error of each classifier tends to increase with each iteration
         24. test error could keep decreasing even after train error converged
         25. normally overfitting doesn't happen unless
             1. too noisy data
             2. so much overlap
         26. train error decreases and exponential error will definetly decrease
         27. solution for overfitting:
             1. stop to much iteration
             2. cross validation
   2. bagging is more robust but boosting more sensitive and may overfit to noise
2. parallel Ensemble methods

   1. bagging tell the definition and formulation , bootstrap aggregation(using random subsets of training data with replacement then voting) use the Breiman 1996 algorith

      1. decreases variance
      2. used on models with low bias and high variance. models that tend to overfit like decision trees
   2. reasons we use bagging on decision trees:

      1. simple models
      2. low sensitivity on noise due to Information gain criteria
      3. low bias, high variance
   3. what is the garantee that bagging will improve performance? classifiers must be diverse so in case one performs poorly the others compensate.

      1. diversification methods:
         1. different data sets
         2. unstable classifiers [! define stable and unstable , unstable is the classifier that doesn't change it's answer too much]
         3. Decision trees are stable classifiers, because of IG.
         4. random forrest: using different set of dimensions for each classifier with is less than square root of the total dimensions
            1. give complete random forest alforithm for T subtrees with m dimensions less than D
            2. drawing bootsrap in random forest is about both datas and features.
            3. sum voting for declaring final label
   4. we use same weights for the bagging (but for boosting it's considered different)
