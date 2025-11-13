import os
import sys
import torch
from torch.utils.data import DataLoader, random_split

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
from torchcp.classification.score import LAC
from torchcp.classification.predictor import SplitPredictor