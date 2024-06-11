import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms.functional
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DementiaDataset(Dataset):
    """
    Initialize the Dataset class passing the arguments we need:
        - csv_file: The file with the labels and relative paths
        - dir_to_images: A directory with all the images
        - transform: Any transformations to be applied to the image
    """
    def __init__(self, csv_file, dir_to_images, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.dir_to_images = dir_to_images
        self.transform = transform

    """
    Set the lenght method refering to the csv file 
    """
    def __len__(self):
        return len(self.annotations)
    
    """
    We set the getitem function. We need:
        - img_path: The full path to an image. It gets it by accessing the
        first column and iterating over it with the index parameter
        - image: open the image and convert it to greyscale with .convert('L)
        - label: get the label from self.annotations, iterating with index to
        get the right label for each image. Then convert this to a tensor

        If a transform method is provided, it also trnasforms the image.
    """
    def __getitem__(self, index):
        img_path = os.path.join(self.dir_to_images,
                                self.annotations.iloc[index, 1])
        image = Image.open(img_path).convert('L')
        label = torch.tensor(int(self.annotations.iloc[index, 2]))

        if self.transform:
            image = self.transform(image)

        return image, label
    
    """
    We now make a static method to get mean and standard deviation in case we 
    need to normalize the images when using the transform method.
    """
    @staticmethod
    def get_mean_and_std(dataset):
        tensors = []
        for img, _ in dataset:
            if not torch.is_tensor(img):
                img = torchvision.transforms.functional.to_tensor(img)
            tensors.append(img)

        all_tensors = torch.stack(tensors)
        mean, std = torch.mean(all_tensors), torch.std(all_tensors)

        return mean, std
    
    """
    This is a static method to get the image from a tensor. First, we
    create an instance of ToPILImage(), then we use this to convert a
    tensor (provided in the variable) into a PIL image.
    """
    @staticmethod
    def tensor2img(tensor):
        to_img = transforms.ToPILImage()
        img = to_img(tensor)
        return img
    

if __name__ == "__main__":
    """
    We set up the transformation method. Initially, comment this out and
    uncomment when the mean and std values are computed. Then copy paste 
    in the Normalize() function
    """
    img_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.2786], [0.3268])
    ])

    """
    Initialize the Dataset and pass the paths. FIrst you will need to set 
    trnasform to None since the mean and std are computed later on. After
    that, when the img_transform has the std and mean, change it to the 
    img_transform
    """
    dataset = DementiaDataset(csv_file='labels.csv',
                              dir_to_images='all_images',
                              transform=img_transform)
    
    """
    Use the static method to get the mean and std of the dataset, then comment it
    """
    # mean, std = DementiaDataset.get_mean_and_std(dataset)

    """"
    We use the dataset object to smaple an image and its label. Then we use the 
    image2tensor static method from our Dataset class and plot it with matplotlib.
    The title of the plot will be its class.
    """
    img_tensor, label = dataset[3220]
    img = DementiaDataset.tensor2img(img_tensor)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f'Label: {label.item()}')
    plt.show()
