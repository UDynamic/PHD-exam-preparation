subjects: {bayes classifier and risk matrix (definition, formulation and everything)}
more specifically add flashcards on these notes too [ Don't miss on anything] :

1. fundamental formulas of bayes theorem for C_k classes of data
   1. posterior is likelihood times prior on evidence
2. minimum posterir is error
3. if don't have prior p(c_k) then compare lilkelihoods p(x|c_k). [! why]
4. if don'thave the prior the evidence p(x) is not concidered in the posterior formula for p(c_k|x) = p(x|c_k). [! why and how]
5. what is risk matrix and it's use case.
6. for a binary classification : p(x|c_1) = 1 - p(x|c_2) [! generalize on this]
7. for decision boundary : p(c_1|x) = p(c_2|x) \rightarrow p(x|c_1)p(c_1) = p(x|c_2)p(c_2)
8. for binary classification: if p is the probability of x belonging to c_1 [! clarify on which one of the posterior or likelihood is meant here] then 1-p is the probability of x belonging to c_2 and for p>1-p according to bayes x belongs to c_1 and the error is 1-p
   1. for more than 2 classes x belongs to c_1 if p_{c_1} > \sum p_{c_n} for all other classes and the error will be that exact sum over ps
9. total error for the bayes classifier for binary class is min{p(c_1), p(c_2)} \times (1 - min{p(c_1), p(c_2)}) [! what does it mean and clarify on it]
