import os
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.io import read_image
from torchvision.transforms import Grayscale,Resize
import torch
from torch.utils.data import Dataset, DataLoader

SPLIT= {
    "train" : (0,4980),#80%
    "val" : (4980,5914),#15%
    "test" : (5914,6225),#5%
}


class RoadDataset(Dataset):
    def __init__(self, root, split='train'):
        self.split = split
        self.root = root
        self.meta_data = pd.read_csv(os.path.join(root,"metadata.csv")).dropna() #Use only the data with groundtruch label
        start,end=SPLIT[split]
        self.meta_data=self.meta_data.iloc[start:end]
        
        # Add resizing transformation
        self.resize = Resize((256, 256))
    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        img_metadata= self.meta_data.iloc[idx]
        
        img = read_image(os.path.join(self.root,img_metadata.sat_image_path)).float()  
        img = img / 255.0
        img = self.resize(img)  # Resize to 512x512

        mask = read_image(os.path.join(self.root,img_metadata.mask_path)).float()
        grayscale_transform = Grayscale(num_output_channels=1)
        mask = grayscale_transform(mask)
        mask = mask / 255.0
        mask = (mask > 0.5).long() 
        mask = self.resize(mask)  # Resize to 512x512

        return img, mask
    
if __name__ == '__main__':
    # Load the dataset and data loader
    root='deepglobe-road-extraction-dataset'
    dataset = RoadDataset(root, "train")
    print(f"Total images in dataset: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=1)


    img_source,mask= next(iter(loader))
    # REmove batch number, put channel last
    img_source = img_source.squeeze(0).permute(1,2,0)
    mask = mask.squeeze(0).permute(1,2,0)
    # Plot source and target image
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img_source, cmap='gray')
    axes[0].set_title("Source Image")
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Mask Image")
    axes[1].axis('off')

    plt.show()
