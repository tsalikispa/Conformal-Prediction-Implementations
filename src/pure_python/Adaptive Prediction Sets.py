#Provides functions ofr interacting with the operating system
import os
#Allows you to read and write .json files
import json
#Imports NumPy for numerical computations
import numpy as np
#Used for plotting graphs and images
import matplotlib.pyplot as plt
#allows you to load image files into arrays
from matplotlib.image import imread

if not os.path.exists('../../data'):
    #if the folder ../data doesn't exist runs the comman to download a file from Google Drive
    os.system('gdown 1h7S6N_Rx7gdfO3ZunzErZy6H7620EbZK -O ../data.tar.gz')
    #extracts the .tar.gz archive into the parent directory (../)
    os.system('tar -xf ../data.tar.gz -C ../')
    #Removes the downloaded compressed file to save space
    os.system('rm ../data.tar.gz')

if not os.path.exists('../../data/imagenet/human_readable_labels.json'):
    os.system('wget -nv -O ../data/imagenet/human_readable_labels.json -L https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json')

# Loads a NumPy compressed archive (.npz) containing precomputed results (from a ResNet-152 model on ImageNet).
#It contains arrays like :
#smx:softmax outputs(model confidence scores)
#labels:true class indices
data = np.load('../../data/imagenet/imagenet-resnet152.npz')

#List all files in the folder ../data/images/examples, contiang sample images
example_paths = os.listdir('../../data/imagenet/examples')

#########################Extract Arrays#########################
#The softmax predictions of 50000 samples,1000 probabilities per image
smx = data['smx'] #Array of shape (50000, 1000)
#The ImageNet class indices
labels = data['labels'].astype(int) # (50000,)


#########################Setup#########################
# Number of calibration points
n = 1000
#(1-alpha) is the desired coverage
alpha = 0.1 #Defines a significance level (so the confidence level is (1 - alpha)= 0.9 or 90%

#########################Split data into calibration and validation sets#########################
# We construct a boolean mask ( an array of True/False values ) that help us pick randomly n samples(calibration set) and use
#the remaining samples as validation set
idx = np.array([1]*n+[0]*(smx.shape[0]-n)) >0
#Randomly shuffle the boolean arrays so the calibration and validation sets are randomly distributed
np.random.shuffle(idx)
# We use the boolean mask to select rows
cal_smx,val_smx = smx[idx, :], smx[~idx,:] #Softmax scores for calibration and validation samples
cal_labels, val_labels = labels[idx], labels[~idx] #True labels for calibration and validation samples

################# Conformal Prediction##########################
#cal_smx--> (n,100), have the softmax scores for each image across 1000 ImageNet classes.
#.argsort(1) sorts the indices of the probabilities along each row
#[:.::-1] reverses the order->now sorted descending
#cal_pi[i] give the class indices ordered from most likely to least likely for sample i.
cal_pi = cal_smx.argsort(1)[:,::-1]

# Now we use the indices from cal_pi in order to rearange the cal_smx softmax values in descending order

cal_srt = np.take_along_axis(cal_smx,cal_pi,axis=1).cumsum(axis=1) #Gives the cumulative probability mass as you add more classes

# We calculate the cumulative softmax mass required to include the true label for image i.
cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
    range(n), cal_labels
]
# We calculate the cutoff cumulative probability that ensures 90 % of true labels are within prediction sets
#interpolation ='higher' means it picks the next-highest score above that quantile, ensuring coverage is not underestimated.
qhat = np.quantile(
    cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher"
)
# Sorts the softmax probabilities in descending order per sample for the validation set.
#val_pi indices of classes ordered from most to least probable
val_pi = val_smx.argsort(1)[:, ::-1]
#Sorts and cumulatively sums softmax socres for validation samples
val_srt = np.take_along_axis(val_smx, val_pi, axis=1).cumsum(axis=1)
# Now we build the prediction sets for each validation sample
#val_srt <= qhat → Boolean array marking which cumulative probabilities are ≤ threshold qhat.
# val_pi.argsort(axis=1) Inverse permutation to map back to original class indices.
prediction_sets = np.take_along_axis(val_srt <= qhat, val_pi.argsort(axis=1), axis=1)

################Compute Empirical Coverage#############################
#Selects, for each validation sample , whether the true label as inside its prediction set (True or False)
#and returns the fraction of validation samples where the true label was included
empirical_coverage = prediction_sets[np.arange(prediction_sets.shape[0]), val_labels].mean()
print(f"The empirical coverage is: {empirical_coverage}")

###################Load readable ImageNet labels###########################
#Opens the JSON file containing human-readable ImageNet class names (e.g., “zebra”, “coffee mug”).
#Loads it into a Python list, then converts to a NumPy array for easier indexing.
with open('../../data/imagenet/human_readable_labels.json') as f:
    label_strings = np.array(json.load(f))

#####################List example image files##########################################
#List all example image filenames in the folder
example_paths = os.listdir('../../data/imagenet/examples')

#########################Show some example predictions###############################
#Loops through 10 random images
#np.random.choice(example_paths) → randomly selects a file name.
#imread(...) → loads the image into memory for display.
#img_index = int(rand_path.split('.')[0]) → extracts the numeric index from the filename
#(e.g., "123.jpg" → 123), which is used to access the corresponding row in smx.
#prediction_set = smx[img_index] > 1 - qhat → determines which labels to include in the prediction set for this image.
#Displays the image using Matplotlib (no axis ticks).
#Prints the names of all labels that were included in the prediction set:
# Show some examples
with open("../../data/imagenet/human_readable_labels.json") as f:
    label_strings = np.array(json.load(f))

example_paths = os.listdir("../../data/imagenet/examples")
for i in range(10):
    rand_path = np.random.choice(example_paths)
    img = imread("../data/imagenet/examples/" + rand_path)
    img_index = int(rand_path.split(".")[0])
    img_pi = smx[img_index].argsort()[::-1]
    img_srt = np.take_along_axis(smx[img_index], img_pi, axis=0).cumsum()
    prediction_set = np.take_along_axis(img_srt <= qhat, img_pi.argsort(), axis=0)

    # Get label names for the prediction set
    pred_labels = label_strings[prediction_set]

    # Show image
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis("off")

    # Add labels on the image
    text = "\n".join(pred_labels)
    plt.gcf().text(
        0.02, 0.02, text, fontsize=9, color="white",
        bbox=dict(facecolor="black", alpha=0.6, boxstyle="round,pad=0.4")
    )

    plt.title("Prediction Set", fontsize=12, color="white", pad=10)
    plt.show()








