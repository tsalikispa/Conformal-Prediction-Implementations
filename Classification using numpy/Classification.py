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

if not os.path.exists('../data'):
    #if the folder ../data doesn't exist runs the comman to download a file from Google Drive
    os.system('gdown 1h7S6N_Rx7gdfO3ZunzErZy6H7620EbZK -O ../data.tar.gz')
    #extracts the .tar.gz archive into the parent directory (../)
    os.system('tar -xf ../data.tar.gz -C ../')
    #Removes the downloaded compressed file to save space
    os.system('rm ../data.tar.gz')

if not os.path.exists('../data/imagenet/human_readable_labels.json'):
    os.system('wget -nv -O ../data/imagenet/human_readable_labels.json -L https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json')

# Loads a NumPy compressed archive (.npz) containing precomputed results (from a ResNet-152 model on ImageNet).
#It contains arrays like :
#smx:softmax outputs(model confidence scores)
#labels:true class indices
data = np.load('../data/imagenet/imagenet-resnet152.npz')

#List all files in the folder ../data/images/examples, contiang sample images
example_paths = os.listdir('../data/imagenet/examples')

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

######################Compute conformal scores###########################
#cal_scores measures how "wrong" the model was for the calibration sample
#np.arange(n):generates an array [0,1,2,....,n-1] one index for each calibration sample
cal_scores = 1 - cal_smx[np.arange(n), cal_labels] # Selects the predicted probability that the model assigned to the true label for each calibration sample.

##############################Compute the adjusted quantile#################################
# We compute the quantile level to achieve the desired coverage
#(1 - alpha) is the desired coverage
#(n+1)*(1-alpha) gives the rank position in sorted scores
q_level = np.ceil((n+1)*(1-alpha)) / n
################################# Compute the Quantile Value###########################
#We calculate qhat such that 90 % of calibration samples have cal_score <=qhat
#We take the quantile(threshold) of the calibration scores at level q_level
#interpolation ='higher' means it picks the next-highest score above that quantile, ensuring coverage is not underestimated.
qhat = np.quantile(cal_scores, q_level, interpolation='higher')

###############Form the predictions sets#########################
# val_smx : softmax probabilities on the validation set
# We create a boolean matrix of shape (num_val_samples,num_class) ,
#where True means the model is confident enough to include that class in the prediction set
#and False means exclude that class
#For each validation image, prediction_sets[i] contains all labels whose probability is high enough
prediction_sets = val_smx >= (1 - qhat)

###############################Compute Empirical Coverage#############################
#Selects, for each validation sample , whether the true label as inside its prediction set (True or False)
#and returns the fraction of validation samples where the true label was included
empirical_coverage = prediction_sets[np.arange(prediction_sets.shape[0]), val_labels].mean()
print(f"The empirical coverage is: {empirical_coverage}")

###################Load readable ImageNet labels###########################
#Opens the JSON file containing human-readable ImageNet class names (e.g., “zebra”, “coffee mug”).
#Loads it into a Python list, then converts to a NumPy array for easier indexing.
with open('../data/imagenet/human_readable_labels.json') as f:
    label_strings = np.array(json.load(f))

#####################List example image files##########################################
#List all example image filenames in the folder
example_paths = os.listdir('../data/imagenet/examples')

#########################Show some example predictions###############################
# Loops through 10 random images
# np.random.choice(example_paths) → randomly selects a file name.
# imread(...) → loads the image into memory for display.
# img_index = int(rand_path.split('.')[0]) → extracts the numeric index from the filename
# (e.g., "123.jpg" → 123), which is used to access the corresponding row in smx.
# prediction_set = smx[img_index] > 1 - qhat → determines which labels to include in the prediction set for this image.
# Displays the image using Matplotlib (no axis ticks).
# Prints the names of all labels that were included in the prediction set:
for i in range(10):
    rand_path = np.random.choice(example_paths)
    img = imread('../data/imagenet/examples/' + rand_path)
    img_index = int(rand_path.split('.')[0])

    # The softmax probabilities for a single image (NumPy array length 1000)
    probs = smx[img_index]

    # Gives a boolean mask something like prediction_set=[False, False, True, True, False,...],
    # True means include class in the prediction set
    prediction_set = probs > 1 - qhat

    # Get the labels and probabilities for those classes
    selected_labels = label_strings[prediction_set]
    selected_probs = probs[prediction_set]

    # Sort by probability (descendin

for i in range(10):
    rand_path = np.random.choice(example_paths)
    img = imread('../data/imagenet/examples/' + rand_path)
    img_index = int(rand_path.split('.')[0])

    # Softmax probabilities for this image
    probs = smx[img_index]

    # Boolean mask for prediction set
    prediction_set = probs > 1 - qhat

    # Extract labels and probabilities for selected classes
    selected_labels = label_strings[prediction_set]
    selected_probs = probs[prediction_set]

    # Sort by probability (descending)
    sorted_idx = np.argsort(selected_probs)[::-1]
    selected_labels = selected_labels[sorted_idx]
    selected_probs = selected_probs[sorted_idx]

    # Combine labels and probabilities into strings
    predicted_labels = [
        f"{label}: {p * 100:.1f}%" for label, p in zip(selected_labels, selected_probs)
    ]

    # Limit display to top 5 for readability
    top_labels = predicted_labels[:5]

    # --- Plot ---
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis('off')

    # Add label text box on image
    text = "\n".join(top_labels)
    plt.gcf().text(
        0.02, 0.02, text,
        fontsize=8, color='white',
        bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.4')
    )

    plt.title("Prediction Set", fontsize=10, color='white', pad=8)
    plt.show()



