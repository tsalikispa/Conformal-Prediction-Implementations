#matplotlib.pyplot --> used for plotting training curves
import matplotlib.pyplot as plt
#DropoutMLP → Our neural network Class
from DropoutMLP import DropoutMLP
# FashionMNIST → our dataset class that creates DataLoaders.
from FashionMNIST import FashionMNIST
# Trainer - > Our training loop Class
from trainer import Trainer
import torch
""" The hyper parameters for our model:
     - num_outputs=10: FashionMNIST has 10 classes
     - num_hiddens_1=256: First hidden layer has 256 neurons
     - num_hiddens_2=256: Second hidden layer also 256
     - dropout_1=0.5: Drop 50 % of neurons in hidden layer 1
     - dropout_2=0.5 -> drop 50% in hidden layer 2
     - lr=0.1 ->learning rate"""

hparams = {'num_outputs':10, 'num_hiddens_1':256, 'num_hiddens_2':256,
           'dropout_1':0.5, 'dropout_2':0.5, 'lr':0.1}
# We create the model
model = DropoutMLP(**hparams)
# We load FahsionMNIST dataset (train+test)
# We resize and transform and store the batches
data = FashionMNIST(batch_size=256)
# We create a DataLoader for training, shuffle and load the data in batches of 256
train_loader=data.get_dataloader(True)
# We create a DataLoader for validation , we do not shuffle the data
val_loader=data.get_dataloader(False)
# We create the trainer that will run the trainingloop for 10 epochs.
# It tracks training losses, validation losses, training accuracy and validation accuracy
trainer = Trainer(epochs=10)

"""" For each epoch:

1. Training loop
    - Forward pass
    - Compute loss
    - Backprop
    - Update weights
    - Track training loss and accuracy
2.Validation loop (eval)
    - Call trainer.eval(...)
    - No gradients
    - Compute validation loss & accuracy
    - Track validation metrics """

trainer.fit(model,train_loader,val_loader)
# Saving the Model for later use
# This saves all the learnd parameters: weight matrices, biases etc.
torch.save(model.state_dict(), "fashion_mnist_mlp.pth")
print("Model saved!")

# We plot training and validation metrics
plt.plot(range(1,len(trainer.train_losses)+1),trainer.train_losses,label="Train Loss")
plt.plot(range(1,len(trainer.train_losses)+1),trainer.evaluation_losses, label="Eval Loss")
plt.plot(range(1,len(trainer.val_acc)+1),trainer.val_acc, label="Eval Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("training vs Evaluiation Loss")
plt.legend()
plt.show()


