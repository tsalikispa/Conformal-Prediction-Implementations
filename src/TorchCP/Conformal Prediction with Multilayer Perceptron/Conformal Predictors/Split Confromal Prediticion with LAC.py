import os
import sys
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
# 1. Compute the path to "Dropout MLP Concise Model"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(
    SCRIPT_DIR,
    "..",                    # go up into "Conformal Prediction with Multilayer Perceptron"
    "Dropout MLP Concise Model"
)
PROJECT_ROOT = os.path.normpath(PROJECT_ROOT)

# 2. Add that directory to Python's import search path
sys.path.append(PROJECT_ROOT)

# 3. Now we can import our modules there
from DropoutMLP import DropoutMLP
from FashionMNIST import FashionMNIST
from trainer import Trainer

# 4. TorchCP imports
# We imort the Least Ambiguous Classifier score funstion for confromal prediction
from torchcp.classification.score import LAC
# We import the SplitPredictor in order to implement Split COnfromal Prediction or ICP
from torchcp.classification.predictor import SplitPredictor

##############We Prepare teh dataset and model#####################

# Hyperparameters MUST match those used during training
hparams = {
    'num_outputs': 10,
    'num_hiddens_1': 256,
    'num_hiddens_2': 256,
    'dropout_1': 0.5,
    'dropout_2': 0.5,
    'lr': 0.1,
}
### CLASS_NAMES
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# 1. Create model and load trained weights
model = DropoutMLP(**hparams)
# We load our saved gained through the MLP training procedure
model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, "fashion_mnist_mlp.pth")))
# 2. Prepare FashionMNIST dataset
data = FashionMNIST(batch_size=256)

# We use the test split (data.val) for calibration + test
full_test_dataset = data.val           # torchvision FashionMNIST test set (10k samples)
n_total = len(full_test_dataset)
n_cal   = n_total // 2                 # 5000 for calibration
n_test  = n_total - n_cal              # 5000 for test

cal_dataset, test_dataset = random_split(full_test_dataset, [n_cal, n_test])

# 3. Build dataloaders for TorchCP
cal_dataloader  = DataLoader(cal_dataset,  batch_size=256, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# 4. Preparing a pytorch model for inference & CP
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # VERY important

# 5. Define the conformal prediction algorithm
predictor = SplitPredictor(
    score_function=LAC(),  # you can also try APS, SAPS, RAPS
    model=model,
    alpha=0.1,             # target coverage ≈ 90%
    device=device,
)
# 6. Calibrating the predictor
predictor.calibrate(cal_dataloader)

# Predict for a batch
test_instances, test_labels = next(iter(test_dataloader))
test_instances = test_instances.to(device)
prediction_sets = predictor.predict(test_instances)
# Convert tensors → numpy for convenience
prediction_sets = prediction_sets.cpu().numpy()
print("\n============================")
print(" Prediction Sets (first 20) ")
print("============================\n")

for i in range(20):  # first 5 test samples
    class_indices = np.where(prediction_sets[i] == 1)[0]  # classes included in set
    selected_labels = [class_names[idx] for idx in class_indices]
    print(f"Test sample {i}: Prediction set -> {selected_labels}")
# Evaluate CP performance
result_dict = predictor.evaluate(test_dataloader)
print(f"Coverage Rate: {result_dict['coverage_rate']:.4f}")
print(f"Average Set Size: {result_dict['average_size']:.4f}")
