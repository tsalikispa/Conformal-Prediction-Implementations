# We import pytorch so we can use torch.no_grad() during evaluation
import torch


class Trainer:
    """This class encapsulates the training loop, the evaluation loop and the tracking of losses and accuracies"""
    def __init__(self,epochs):
        # In the constructor we save the number of epochs and
        # we initialize lists to track training losses, validation losses, training accuracies and validation accuracy
        self.epochs = epochs
        self.train_losses=[]
        self.evaluation_losses=[]
        self.train_acc=[]
        self.val_acc=[]
    def fit(self,model,data, val_data=None):
        """model: The neural network
           data: training dataloader
            val_data: optional validation dataloader"""
        # We set model to training mode
        # This is crucial in order individual layers to behave in training mode
        model.train()
        # Our model has a method that returns an optimizer (SGD).
        # This builds the optimizer that will update the model weights
        optimizer=model.configure_optimizers()

        # The training repeats for sel.epochs iterations though the entire dataset
        for epoch in range(self.epochs):
            #We initialize epoch metrics.
            #These accumulate loss and accuracy over the whole epoch.
            epoch_loss=0
            epoch_acc=0

            #Mini-batch training loop
            # At each iteration we receieve a minibatch:
            # X: batch of images, shape ( batch,channels, H, W)
            # y: labels
            for X,y in data:
                # We don't want PyTorch to accumulate gradients across batches so we have to clear the gradient before computing new ones
                optimizer.zero_grad()
                # Compute logits and pass through our constructed Multilayer Perceptron
                y_hat=model.forward(X)
                # We use our model's built-in loss function (CrossEntropyLoss)
                minibatch_loss=model.loss(y_hat,y)
                #We implement backpropagation. We compute the gradients and store them inside each parameter's .grad field
                minibatch_loss.backward()
                # We apply gradient descent update
                optimizer.step()
                # We accumulate metrics
                epoch_loss+=minibatch_loss.item() # .item() extracts the Python value of the tensor loss.
                epoch_acc+=model.accuracy(y_hat,y)
            #We compute average metrics for the epoch.
            # We have summed the loss/accuracy over all minibatches. Now we average them per epoch.
            epoch_loss = epoch_loss/len(data)
            epoch_acc = epoch_acc/len(data)
            # We save the metrics
            self.train_losses.append(epoch_loss)
            self.train_acc.append(epoch_acc)
            # If we have validation data, we evaluate the model after each epoch
            if val_data is not None:
                self.eval(model,val_data)
            print(f"Epoch {epoch+1}: Loss {epoch_loss:.4f} | Acc {epoch_acc*100:.2f}%")



    def eval(self,model,data):
        """This is the evaluation function. """
        # Switches the model to evaluation mode:
        #         - dropout-->disabled
        model.eval()

        evaluation_loss=0
        evaluation_acc=0
        # Disable gradient computation
        with torch.no_grad():
            # We loop through vlalidation data
            # No gradient steps, only forward passes
            for X,y in data:
                y_hat=model(X)
                loss=model.loss(y_hat,y)
                # Average validation metrics
                evaluation_loss+=loss.item()
                evaluation_acc+=model.accuracy(y_hat,y)
        # Save metrics
        evaluation_acc=evaluation_acc/len(data)
        evaluation_loss=evaluation_loss/len(data)
        self.evaluation_losses.append(evaluation_loss)
        self.val_acc.append(evaluation_acc)
        print(f"Validation: Loss {evaluation_loss:.4f} | Acc {evaluation_acc*100:.2f}%")








