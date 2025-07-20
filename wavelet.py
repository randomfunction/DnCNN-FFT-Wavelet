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

class WaveletBlock(nn.Module):
    """
    Predicts the noise residual in the wavelet domain.
    Uses DWT (Discrete Wavelet Transform) to analyze image in frequency + spatial domain.
    """
    def __init__(self, channels, wavelet='haar'):
        super().__init__()

        self.dwt = DWT(J=1, wave=wavelet)
        self.idwt = IDWT(wave=wavelet)

        self.ffn = nn.Sequential(
            nn.Conv2d(4 * channels, 4 * channels, kernel_size=1, padding=0),  
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * channels, 4 * channels, kernel_size=1, padding=0)   
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


class HybridDnCNN(nn.Module):
    """
    Combines standard DnCNN residual prediction with a wavelet-based residual branch.
    Input: noisy image x
    Output: denoised image
    """
    def __init__(self, channels=3, num_layers=17, features=64, wavelet='haar'):
        super(HybridDnCNN, self).__init__()
        layers = [
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        ]
        for _ in range(num_layers - 2):
            layers += [
                nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
            ]
        layers.append(nn.Conv2d(features, channels, kernel_size=3, padding=1, bias=False))
        self.dncnn = nn.Sequential(*layers)

        # Wavelet residual branch
        self.wavelet_block = WaveletBlock(channels, wavelet)

    def forward(self, x):
        noise_dn = self.dncnn(x)
        noise_wave = self.wavelet_block(x)
        noise_combined = noise_dn + noise_wave
        clean = x - noise_combined
        return clean
    

model = HybridDnCNN(channels=3).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 100
noise_std = 0.1 

print("Starting Training...")
model.train()  
best_loss = float('inf')
patience = 2  
patience_counter = 0
wavelet_state = None


for epoch in range(num_epochs):
    epoch_loss = 0
    start_time = time.time()

    for data,_ in train_loader:
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
        wavelet_state = model.state_dict() 
    else:
        patience_counter += 1
        print(f"No improvement. Patience: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

if wavelet_state is not None:
    model.load_state_dict(wavelet_state)
    print("Loaded best model with lowest validation loss.")


if wavelet_state is not None:
    model.load_state_dict(wavelet_state)
    print("Loaded best model with lowest validation loss.")
    torch.save(model.state_dict(), 'wavelet_model.pth')
    print("Best model saved as 'wavelet.pth'.")



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