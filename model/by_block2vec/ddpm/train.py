import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import wandb

from ddpm import DDPM
from model import UNet3D

class Random3DDataset(Dataset):
    def __init__(self, npy_file_path):
        super(Random3DDataset, self).__init__()
        numpy_data = np.load(npy_file_path)
        self.data = torch.from_numpy(numpy_data).float()
        self.num_samples = self.data.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        sample = self.data[index]
        return sample

def train_ddpm(model, dataloader, optimizer, device, epochs=10, project_name="ddpm_3d"):

    wandb.init(project=project_name, config={
        "epochs": epochs,
        "learning_rate": optimizer.param_groups[0]['lr'],
        "batch_size": dataloader.batch_size,
        "T_diffusion": model.n_T,
        "architecture": model.eps_model.__class__.__name__,
        "base_filters": model.eps_model.inc.double_conv[0].out_channels if hasattr(model.eps_model, 'inc') else "N/A", # 尝试获取 base_filters
        "dataset_size": len(dataloader.dataset)
    })

    wandb.watch(model, log="all", log_freq=100)

    model.train()
    global_step = 0
    for epoch in range(epochs):
        running_loss = 0.0
        running_100_loss = 0.0
        for i, sample in enumerate(dataloader):
            sample = sample.to(device)
            optimizer.zero_grad()
            loss = model(sample)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            running_loss += current_loss

            wandb.log({"epoch": epoch + 1, "step_loss": current_loss, "global_step": global_step})
            global_step +=1
            running_100_loss += current_loss
            if (i + 1) % 10 == 0:
                avg_loss = running_loss / 10
                wandb.log({"epoch_avg_loss_per_10_steps": avg_loss, "epoch": epoch + 1, "batch_step": i+1})
                print(f"Epoch [{epoch+1}/ {epochs}], Step [{i+1}/{len(dataloader)}], Loss: {avg_loss:.4f}")
                running_loss = 0.0

                if (i + 1) % 500 == 0:
                    avg_100_loss = running_100_loss / 500
                    wandb.log({"epoch_avg_loss_per_100_steps": avg_100_loss, "epoch": epoch + 1, "batch_step": i+1})
                    running_100_loss = 0.0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    npy_data_file_path = "transformed_embedded_dataset.npy"

    T = 4000
    
    eps_model = UNet3D(time_channels=256, block_embedding_dimensions=6)
    criterion = nn.MSELoss()
    ddpm_model = DDPM(eps_model=eps_model, n_T=T, criterion=criterion)
    ddpm_model.to(device)
    
    optimizer = optim.Adam(ddpm_model.parameters(), lr=2e-4)
    
    dataset = Random3DDataset(npy_file_path=npy_data_file_path)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    
    epochs = 10
    train_ddpm(ddpm_model, dataloader, optimizer, device, epochs)
    torch.save(ddpm_model.state_dict(), "ddpm_final.pth")

    wandb.finish()

if __name__ == "__main__":
    main()