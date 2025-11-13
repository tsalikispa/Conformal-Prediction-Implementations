#torch.utils.data--> Pytorch's module for handling datasets and dataloaders
import torch.utils.data
#Dataset--> Base class that we subclass to create custom datasets
# Dataloader--> Creates mini-batches, shuffling, parallel loading
# TensorDataset--> A simple dataset wrapping tensor
from torch.utils.data import Dataset,DataLoader,TensorDataset
# Torchvision--> Library that contains ready-to-use vision datasets like FashionMnist
import torchvision
#transforms -->Preprocessing tools (resize, normalization, convert to tensor)
from torchvision import transforms
# Used to get the current working directory
import os
# from showimages import show_images


class FashionMNIST(Dataset):
    """We create our own dataset class that inherits from Pytorch's Dataset. This class is not the data itself.
     It's a wrapper that:
     - Loads the torchvision FashionMNIST dataset
     - Applies Transformations
     - Creates its own dataloaders"""
    def __init__(self,batch_size=64,resize=(28,28)):
        # The constructor takes the batch_size (default mini-batch size) and resize (default image size)
        super().__init__()
        #os.getcwd() gives the folder where the script is running. This folder will be user to download the dataset
        self.root=os.getcwd()
        # transforms.compose([])-->Creates a pipeline of image transformations
        # transforms.Resize(resize) --> ensures every image is resized (28x28)
        # transforms.ToTensor() --> converts an image to a tensor
        trans=transforms.Compose([transforms.Resize(resize),transforms.ToTensor()])
        # We load the training set
        self.train=torchvision.datasets.FashionMNIST(root=self.root,train=True,transform=trans,download=True)
        # We load the validation set
        self.val=torchvision.datasets.FashionMNIST(root=self.root,train=False,transform=trans,download=True)
        # This is the batch_size
        self.batch_size=batch_size


    def text_labels(self,indices):
        """FashionMNIST labels are integers 0-9. This function maps them to strings
           This function takes a list/tensor of label indices , converts each index to astring and returns a python list of names"""
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                  'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

        return [labels[int(i)] for i in indices]

    def get_dataloader(self,train):
        """This method builds a DataLoader. if train=True we use training data. Otherwise we use test/validation data
        Dataloader:
        - data: dataset to load from
        - self.batch_size: mini-batch size
        - shuffle=train .If it is true we shuffle training data otherwise we don't shuffle validation data.
        - num_workers=2 :Loads data using 2 parallel workers ( faster)"""
        data=self.train if train else self.val
        return torch.utils.data.DataLoader(data,self.batch_size,shuffle=train,num_workers=2)

    # def visualize(self,batch,nrows=1,ncols=8,labels=[]):
    #     X,y = batch
    #     if not labels:
    #         labels=self.text_labels(y)
    #         show_images(X.squeeze(1),nrows,ncols,titles=labels)
