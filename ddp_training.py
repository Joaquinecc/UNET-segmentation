
from unet import Unet
from dataset import RoadDataset

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import os
from tqdm import tqdm



def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        max_epochs: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data=val_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.max_epochs=max_epochs
        self.model = DDP(model, device_ids=[gpu_id])
        # Define class weights for the loss function
        self.background_weight = 0.2  # Adjust this based on your dataset
        self.road_weight = 1.0  # Adjust this based on your dataset
        self.weights = torch.tensor([self.background_weight, self.road_weight]).to(self.gpu_id)
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=self.weights)
        self.best_val_loss = float('inf')
    def _run_epoch_train(self, epoch):
        batch_size = len(self.train_data)
        self.train_data.sampler.set_epoch(epoch)
        running_loss=0

        self.model.train()
        iterator = tqdm(self.train_data, desc=f"Epoch [{epoch+1}/{self.max_epochs}] Training", disable=(self.gpu_id != 0))

        for img, mask in iterator:
            img = img.to(self.gpu_id)
            mask = mask.to(self.gpu_id)
            self.optimizer.zero_grad()
            logits = self.model(img)

            # Compute the weighted loss
            loss = self.loss_fn(logits, mask.squeeze(1))
            running_loss += loss.item()
            

            loss.backward()
            self.optimizer.step()
        if self.gpu_id == 0:
            print(f"Epoch [{epoch+1}/{self.max_epochs}] Average Training Loss: {running_loss / batch_size:.4f}")

        
    def _run_epoch_val(self, epoch):
        self.val_data.sampler.set_epoch(epoch)
        val_loss=0
        true_positive = 0
        false_positive = 0
        false_negative = 0

        self.model.eval()
        iterator = tqdm(self.val_data, desc=f"Epoch [{epoch+1}/{self.max_epochs}] Validation", disable=(self.gpu_id != 0))
        with torch.no_grad():  # No need to compute gradients during validation
            for img, mask in iterator:
                img = img.to(self.gpu_id)
                mask = mask.to(self.gpu_id)
                logits = self.model(img)
                
                # Compute the weighted loss
                loss = self.loss_fn(logits, mask.squeeze(1))
                val_loss += loss.item()

                pred_probab = torch.nn.Softmax(dim=1)(logits)
                # Compute accuracy
                preds = torch.argmax(pred_probab, dim=1)  # Get class predictions
                true_positive += ((preds == 1) & (mask.squeeze(1) == 1)).sum().item()
                false_positive += ((preds == 1) & (mask.squeeze(1) == 0)).sum().item()
                false_negative += ((preds == 0) & (mask.squeeze(1) == 1)).sum().item()

        # Compute average loss and accuracy
        avg_loss = val_loss / len(self.val_data)
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0

        if self.gpu_id == 0:
            print(f"Epoch [{epoch+1}/{self.max_epochs}] Validation Loss: {avg_loss:.4f}, Precision: {precision:.4%}")

        return avg_loss
            
    def _save_checkpoint(self, epoch):
        # Ensure the folder exists
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save the checkpoint
        PATH = os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pt")
        ckp = self.model.module.state_dict()
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")
    def _save_best_model(self):
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        PATH = os.path.join(checkpoint_dir, "best_model.pt")
        ckp = self.model.module.state_dict()
        torch.save(ckp, PATH)
        print("Best model saved!")

    def train(self: int):
        for epoch in range(self.max_epochs):
            self._run_epoch_train(epoch)
            val_loss= self._run_epoch_val(epoch)
            if self.gpu_id == 0:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_best_model()

                if epoch % self.save_every == 0:
                    self._save_checkpoint(epoch)

def load_train_objs(checkpoint_path):
    data_path='/ediss_data/ediss6/deepglobe-road-extraction-dataset'
    dataset_train = RoadDataset(data_path, "train")
    dataset_val = RoadDataset(data_path, "val")
    model = Unet(input_channel=3,class_num=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
        print("Checkpoint loaded successfully!")
    return dataset_train,dataset_val, model, optimizer  


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size_train: int,batch_size_val: int,checkpoint_path=None):
    print(f"Rank: {rank}, World Size: {world_size}, Save Every: {save_every}, Total Epochs: {total_epochs}, Batch Size Train: {batch_size_train}, Batch Size Val: {batch_size_val}")
    ddp_setup(rank, world_size)
    dataset_train,dataset_val, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset_train, batch_size_train)
    val_data = prepare_dataloader(dataset_val, batch_size_val)
    trainer = Trainer(model, train_data,val_data, optimizer, rank, save_every,total_epochs)
    trainer.train()
    destroy_process_group()


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description='simple distributed training job')
    # parser.add_argument('total_epochs', type=int,default=10 help='Total epochs to train the model')
    # parser.add_argument('save_every', type=int,default=5 help='How often to save a snapshot')
    # parser.add_argument('--batch_size', default=3, type=int, help='Input batch size on each device (default: 32)')
    # args = parser.parse_args()
    total_epochs=16
    save_every=5
    batch_size_train=24
    batch_size_val=8
    world_size = torch.cuda.device_count()
    checkpoint_path =None
    mp.spawn(main, args=(world_size,save_every, total_epochs,batch_size_train,batch_size_val,checkpoint_path), nprocs=world_size)