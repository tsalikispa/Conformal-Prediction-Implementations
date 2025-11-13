from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
# We use return_X_y=True in order the function to return X->data and y -> the labels instead of a full dictionary
X, y = load_iris(return_X_y=True)
# We save the class names for later use
class_names = load_iris().target_names
'''We keep 60% for training ( X_train, y_train) and we send 40 % to X_temp,y_temp (temporary set for calibariotn + test) '''
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
'''Now we split that 40 % into two halves 20 % calibration and 20 % test '''
X_cal,X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
'''multi_class='miltinomial --> we use the softmax version
    max_iter = 100 --> Means the optimizer can take up to 1000 iterations'''
clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
''' We sove the maximum likelihood estimation (MLE) problem. 
In order to find the parameters that best describe the relationship between the features and labels. '''
clf.fit(X_train, y_train)
probs_cal = clf.predict_proba(X_cal)
probs_test = clf.predict_proba(X_test)
print("Example softmax-like probabilities (first 5):")
print(probs_test[:5])
''' Nonconformity scores on the calibration set
[probs_cal[i, y_cal[i]--> teh predicted probability for the true class of sample i'''
scores = 1 - np.array([probs_cal[i, y_cal[i]] for i in range(len(y_cal))])
print(f"The shape of scores is {scores.shape}")
print("The unonconformity scores for every sample are:")
print()

print(scores)

alpha= 0.1
n = len(scores)
# Theoretical index, it tell you which score in the sorted list of nonconformity scores to pick
k = np.ceil((n+1)*(1-alpha))
""" This represents the fraction of the distribtution below of the desired quantile.
 Dividing by n converts from example the theoritical index 28th of 30--->28/30=0.933.
  which NumPy understands as the 93.33rd percentile"""
q_level = k/n
qhat = np.quantile(scores, q_level, interpolation='higher') #If the quantile index falls between two calibration scores,choose the higher one — not interpolate between them.

print("Sorted scores:", np.sort(scores))
print("Quantile level:", q_level)
#q̂ is the threshold nonconformity score such that approximately (1 − α) = 90% of the calibration scores are smaller than or equal to it
print("q̂ =", qhat)
"""1-qhat--> Is the minimum acceptable probability for inclusion in the prediction set """
for i, probs in enumerate(probs_test):
    #np.where(probs >= (1-qhat)) return the indices of all classes whose predicted probability is greater than 1-q
    selected_classes = np.where(probs >= (1 - qhat))[0] #The [0] at the end extracts the array of indices
    selected_labels = class_names[selected_classes]
    print(f"Test sample {i}: Prediction set -> {selected_labels}")
# We construct a matrix that every row is a boolean mask and contain the True value in the indices that the probabilities are >= 1-q
prediction_sets = probs_test >= (1-qhat)
# We check whether the true class y_test[i] is included (True) or excluded (False) in the prediction set of test sample i.
## Taking the mean of a boolean array in NumPy automatically converts: True -> 1 and False->0
empirical_coverage = prediction_sets[np.arange(prediction_sets.shape[0]),y_test].mean()


print(f"The empirical coverage is: {empirical_coverage}")
