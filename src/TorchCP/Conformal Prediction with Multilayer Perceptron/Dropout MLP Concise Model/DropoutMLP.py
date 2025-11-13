# torch: The core PyTorch library
import torch
# torch.nn is Pytorch's neural network module.
import torch.nn as nn

class DropoutMLP(nn.Module):
    """num_ouputs: Number of classes(for classification)
       num_hiddens_1: Number of neurons in first hidden layer
       num_hiddens_2: Number of neurons in second hidden layer
       dropout_1: dropout rate for first hidden layer (probability of zeroing out a unit)
       dropout_2: The same for second hidden layer
       lr: learning rate for optimizer"""
    def __init__(self,num_outputs, num_hiddens_1, num_hiddens_2,dropout_1,dropout_2,lr):
        # We call the parent class (nn.Module) constructor in order to setup internal machinery
        super().__init__()
        # We store hyperparameters for later use
        self.lr=lr
        self.dropout_1=dropout_1
        self.dropout_2 = dropout_2

        # Loss Module
        self.loss_fn = nn.CrossEntropyLoss()

        # We group layers into noe object. When we call self.net(x), it applies each layer in order
        self.net = nn.Sequential(
            # Turn inputs like (batch size, channels, height,width) (like images) --->(batch_size, features) so it can be fed to linear layers
            nn.Flatten(),
            #A fully connected (dense) layer.
            #Lazy mean PytorCh will infer the input feature size the first time it seed data, and then initialize Weights
            # output shape: (batch_size, num_hiddens_1)
            nn.LazyLinear(num_hiddens_1),
            # Applies the activation function ReLU(x) = max(0,x) element-wise
            # This introduces non-linearity, allowing the model to aproximate more complex functions.
            nn.ReLU(),
            # During training: We randomly set eacfh neuron to zero with probability dropout_1
            # This helps prevent overfitting by maing neurons not rely too heavily on specific other neurons
            nn.Dropout(dropout_1),
            #In the next block we implement the same idea but for second hidden Layer
            nn.LazyLinear(num_hiddens_2),
            nn.ReLU(),
            nn.Dropout(dropout_2),
            # In the final layer we output logits of size num_outpus (one_logit per class_
            nn.LazyLinear(num_outputs)

        )

    def accuracy(self, Y_hat, Y, averaged=True):
        """Compute the number of correct predictions
           Y_hat: predicted logits
           Y : ground truth labels"""
        # Flattens any leadning dimensions into on ebig batch dimension
        Y_hat = Y_hat.reshape((-1,Y_hat.shape[-1]))
        # Y_hat.argmax(axis) finds the index of the maximum logit per sample.That is the predicted class.
        preds = Y_hat.argmax(axis=1).type(Y.dtype)
        # We compare predicted labels with ground truth labels
        # True-->for Correct and Flase--> for Incorrect
        compare = (preds == Y.reshape(-1)) # bool tensor
        # We convert True to 1.0 and Flase to 0.0
        # Now accurace is the mean of this vector
        compare = compare.float()
        # If averaged =True we return a scalar accuracy between 0 and 1
        #If false we return the per sample correctness vector
        return compare.mean() if averaged else compare



    def forward(self,X):
        """This function defined the computational graph of the model. When we call model(x), PyTorch internally calls forward(x)"""
        return self.net(X)


    def loss(self, y_hat, y):
        """ We use the built-in Corss entropy loss function.
        The nn.CVorssEntropyLoss internally :
        1. Plly log-softmax to y_hat(logits)
        2. Selects probability of the correct class
        3 Applies negative log-likelihood"""
        return self.loss_fn(y_hat, y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(),self.lr)