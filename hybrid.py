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
from pytorch_wavelets import DWT, IDWT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

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

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((64,64)) #update
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

batch_size = 128
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_loader  = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

import torch
import torch.nn as nn
from torch.fft import fft2, ifft2
from pytorch_wavelets import DWT, IDWT

class FNetBlock(nn.Module):
    def __init__(self, channels, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)  # GroupNorm
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1)
        )

    def forward(self, x):
        x_fft = fft2(x, dim=(-2, -1))
        x_ifft = ifft2(x_fft, dim=(-2, -1)).real
        x_norm = self.norm(x_ifft)
        return x_ifft + self.ffn(x_norm)



class WaveletBlock(nn.Module):
    def __init__(self, channels, wavelet='haar'):
        super().__init__()
        self.dwt = DWT(J=1, wave=wavelet) 
        self.idwt = IDWT(wave=wavelet)  
        self.ffn = nn.Sequential(
            nn.Conv2d(4 * channels, 4 * channels, kernel_size=1, padding=0),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(4 * channels, 4 * channels, kernel_size=1, padding=0),  
        )
        self.threshold = nn.Parameter(torch.zeros(3, channels, 1, 1))  

    def forward(self, x):
        ll, yh = self.dwt(x)
        detail = yh[0]  
        lh, hl, hh = torch.unbind(detail, dim=2) 
        stacked = torch.cat([ll, lh, hl, hh], dim=1)

        y = self.ffn(stacked)
        c = x.size(1)
        ll2, lh2, hl2, hh2 = torch.split(y, c, dim=1)
        t = torch.sigmoid(self.threshold) 
        lh2 = lh2 * t[0]  
        hl2 = hl2 * t[1] 
        hh2 = hh2 * t[2]  
        y_high = torch.stack([lh2, hl2, hh2], dim=2) 
        out = self.idwt((ll2, [y_high]))
        return out


class DnCNNBranch(nn.Module):
    """
    Standard DnCNN residual noise predictor.
    This is a simple deep neural network that predicts the noise residuals from the input image.
    """
    def __init__(self, channels=3, num_layers=17, features=64):
        super().__init__()
        layers = [nn.Conv2d(channels, features, kernel_size=3, padding=1, bias=True),  
                  nn.ReLU(inplace=True)]  
        for _ in range(num_layers - 2):
            layers += [nn.Conv2d(features, features, 3, padding=1, bias=False),
                       nn.BatchNorm2d(features),
                       nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(features, channels, 3, padding=1, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class HybridTriBranchModel(nn.Module):
    def __init__(self, channels=3, dncnn_layers=17, dncnn_features=64, fnet_blocks=5, wavelet='haar'):
        super().__init__()
        self.dncnn_branch = DnCNNBranch(channels, dncnn_layers, dncnn_features)
        self.fnet_blocks = nn.Sequential(
            *[FNetBlock(dncnn_features) for _ in range(fnet_blocks)]  
        )
        self.wavelet_branch = WaveletBlock(channels, wavelet)
        self.fusion = nn.Conv2d(channels * 3, channels, kernel_size=1)

    def forward(self, x):
        r_dn = self.dncnn_branch(x)
        feat = self.dncnn_branch.net[0:2](x) 
        feat = self.fnet_blocks(feat)
        r_fnet = self.dncnn_branch.net[-1:](feat)
        r_wave = self.wavelet_branch(x)
        r_cat = torch.cat([r_dn, r_fnet, r_wave], dim=1)
        r = self.fusion(r_cat)
        return x - r


model = HybridTriBranchModel(channels=3).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 50
noise_std = 0.1  


print("Starting Training...")
model.train()  
best_loss = float('inf')
patience = 2  
patience_counter = 0
hybrid_state = None


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
        hybrid_state = model.state_dict() 
    else:
        patience_counter += 1
        print(f"No improvement. Patience: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

if hybrid_state is not None:
    model.load_state_dict(hybrid_state)
    print("Loaded best model with lowest validation loss.")


if hybrid_state is not None:
    model.load_state_dict(hybrid_state)
    print("Loaded best model with lowest validation loss.")
    torch.save(model.state_dict(), 'hybrid.pth')
    print("Best model saved as 'hybrid.pth'.")


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

