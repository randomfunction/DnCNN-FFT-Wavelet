import torch                               
import torch.nn as nn                      
import torch.optim as optim                
from torch.utils.data import DataLoader   
import torchvision                  
import torchvision.transforms as transforms  
import numpy as np                       
import cv2                                
from skimage.metrics import structural_similarity as ssim  
import matplotlib.pyplot as plt         
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def calculate_psnr(denoised, ground_truth):
    mse = np.mean((denoised - ground_truth) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 1.0 
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr
from skimage.metrics import structural_similarity as ssim


def calculate_ssim(denoised, ground_truth):
    return ssim(ground_truth, denoised, data_range=ground_truth.max() - ground_truth.min(), win_size=7, channel_axis=-1)


import torch
import torch.nn as nn
from torch.fft import fft2, ifft2

class FNetBlock(nn.Module):
    def __init__(self, channels, num_groups=8):
        super(FNetBlock, self).__init__()

        # Use GroupNorm for flexibility over varying spatial dimensions
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels)

        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1)
        )

    def forward(self, x):
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        x_ifft = torch.fft.ifft2(x_fft, dim=(-2, -1)).real
        x_norm = self.norm(x_ifft)
        x_out = x_norm + self.ffn(x_norm)
        return x_out


class FnetDnCNNResidual(nn.Module):
    def __init__(self, channels=3, num_features=64, num_fnet_blocks=10, num_groups=8):
        super(FnetDnCNNResidual, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(channels, num_features, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.fnet_blocks = nn.Sequential(
            *[FNetBlock(num_features, num_groups=num_groups) for _ in range(num_fnet_blocks)]
        )

        self.tail = nn.Conv2d(num_features, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        features = self.head(x)
        features = self.fnet_blocks(features)
        predicted_noise = self.tail(features)
        denoised = x - predicted_noise
        return denoised

    
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

batch_size = 128
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# model = DnCNN_FFT(channels=3).to(device)
model = FnetDnCNNResidual(channels=3, num_features=64).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 100
noise_std = 0.1  

print("Starting Training with Early Stopping...")
model.train()

best_loss = float('inf')
patience = 2  
patience_counter = 0
fft_state = None

for epoch in range(num_epochs):
    epoch_loss = 0
    start_time = time.time()

    for data, _ in train_loader:
        data = data.to(device)
        noise = torch.randn_like(data) * noise_std
        noisy_data = data + noise

        output = model(noisy_data)
        loss = criterion(output, data)
        epoch_loss += loss.item() * data.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss /= len(train_dataset)
    elapsed = time.time() - start_time
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}, Time: {elapsed:.2f} sec")

    if epoch_loss < best_loss - 1e-6:  
        best_loss = epoch_loss
        patience_counter = 0
        fft_state = model.state_dict() 
    else:
        patience_counter += 1
        print(f"No improvement. Patience: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

if fft_state is not None:
    model.load_state_dict(fft_state)
    print("Loaded best model with lowest validation loss.")


if fft_state is not None:
    model.load_state_dict(fft_state)
    print("Loaded best model with lowest validation loss.")
    torch.save(model.state_dict(), 'fft.pth')
    print("Best model saved as 'fft.pth'.")



from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class ImageFolderNoClass(Dataset):
    def __init__(self, folder_path, transform=None):
        self.file_paths = [os.path.join(folder_path, f) 
                           for f in os.listdir(folder_path) 
                           if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32))
])

# train_dataset = ImageFolderNoClass('./BSD500/train', transform=transform)
# val_dataset   = ImageFolderNoClass('./BSD500/val', transform=transform)
test_dataset  = ImageFolderNoClass('./BSD68/BSD68', transform=transform)

batch_size = 32
# train_loader = DataLoader(train_dataset+val_dataset, batch_size=batch_size, shuffle=True)
# val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model.eval()
psnr_list = []
ssim_list = []



with torch.no_grad():
    for data, _ in test_loader:
        data = data.to(device)
        noise = torch.randn_like(data) * noise_std
        noisy_data = data + noise
        output = model(noisy_data)
 
        output_np = output.cpu().numpy().transpose(0, 2, 3, 1) 
        clean_np  = data.cpu().numpy().transpose(0, 2, 3, 1)
        noisy_np  = noisy_data.cpu().numpy().transpose(0, 2, 3, 1)
        
        for denoised, clean in zip(output_np, clean_np):
            denoised = np.clip(denoised, 0., 1.)
            clean = np.clip(clean, 0., 1.)
            psnr_val = calculate_psnr(denoised, clean)
            ssim_val = calculate_ssim(denoised, clean)
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)

mean_psnr = np.mean(psnr_list)
mean_ssim = np.mean(ssim_list)

print(f"Test PSNR: {mean_psnr:.2f} dB")
print(f"Test SSIM: {mean_ssim:.4f}")