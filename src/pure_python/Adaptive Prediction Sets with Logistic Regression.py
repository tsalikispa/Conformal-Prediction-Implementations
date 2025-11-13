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

#####Adaptive Prediction Sets#############
"""Each row cal_pi[i] holds the class indices ordered fom most to least probable for sample i. """
#argsort(1) sorts along axis=1 (rows, each sample)
#[:,::-1] reverses each row-> now in descending order
cal_pi = probs_cal.argsort(1)[:,::-1] # shape (n_cal, n_classes)

'''np.rake_along_axis(A, indices, axis) rearranges elements of A according to indices along the given axis
cumsum(axis=1) computes the cumulative sum row by row
The resulting matrix cal_srt has the same shape as probs_cal, each row represents cumulative probability mass as we include more classes'''

cal_srt = np.take_along_axis(probs_cal,cal_pi, axis=1).cumsum(axis=1)

"""cal_pi.argsort(axis=1) This inverse the permutation from before. 
cal_pi told us, for each sample, how t go from original order --> sorted order
cal_pi.argsort() gives the reverse mapping, how to go sorted --> original
No each row tells us where each original calss index appears in the sorted ranking
--------------------
np.take_along_axis(cal_srt, cal_pi.argsort(axis=1),axis=1)
This reorders cal_srt back to the original class order (unsorted)
So now each elemnt [i, j] in the result corresponds to the cumulative probability mass (in sorted order) up to the class j
-------------------------
[np.arange(n), y_cal] We use NumPy indexing to exrtact, for each calibration sample
cals_scores -> shape (n_cal,) one scalar per sample. Large scores= true calss was rankes low.
"""
n = len(cal_pi)
cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[np.arange(n),y_cal]
print(f"The shape of scores is {cal_scores.shape}")
print("The unonconformity scores for every sample are:")
print()

print(cal_scores)
####Quantile Computation#############
alpha= 0.1

# Theoretical index, it tell you which score in the sorted list of nonconformity scores to pick
k = np.ceil((n+1)*(1-alpha))
""" This represents the fraction of the distribtution below of the desired quantile.
 Dividing by n converts from example the theoritical index 28th of 30--->28/30=0.933.
  which NumPy understands as the 93.33rd percentile"""
q_level = k/n
qhat = np.quantile(cal_scores, q_level, interpolation='higher') #If the quantile index falls between two calibration scores,choose the higher one — not interpolate between them.


print("Quantile level:", q_level)
#q̂ is the threshold nonconformity score such that approximately (1 − α) = 90% of the calibration scores are smaller than or equal to it
print("q̂ =", qhat)
######Apply to test set#################


"""val_pi:ranking of classes for test samples
   val_srt:cumulative probability mass in that sorted order
   (val_srt <= qhat) gives a boolean mask marking "included" classes in sorted order.
   val_pi.argsort(axis-1) maps back to original label order
   np.take_along_axis(..., axis=1) rearragens the boolean mask back to match original class indices
   The final result is prediction_sets.shape = (n_test, n_classes)"""
val_pi = probs_test.argsort(1)[:, ::-1]
val_srt = np.take_along_axis(probs_test, val_pi, axis=1).cumsum(axis=1)
prediction_sets = np.take_along_axis(val_srt <= qhat, val_pi.argsort(axis=1), axis=1)

for i in range(len(y_test)):
    # Boolean mask for sample i (True = included in APS prediction set)
    mask = prediction_sets[i]

    # Convert to class indices
    selected_classes = np.where(mask)[0]

    # Convert to readable class labels
    selected_labels = class_names[selected_classes]

    print(f"Test sample {i}: APS prediction set -> {selected_labels}")

##########Empirical Coverage#################


# We check how often the true class of each test samples was included in its prediction set.
# prediction_sets[np.arange(len(y_test)), y_test] selects the boolean entry for each test sample's true label
# .mean() averages the True/False values(True->1, False->0)
empirical_coverage = prediction_sets[np.arange(len(y_test)), y_test].mean()



print(f"The empirical coverage is: {empirical_coverage}")

