import torch
import torch.nn as nn
import torch.nn.functional as F

class ReflectiveAutoencoder(nn.Module):
    def __init__(self):
        super(ReflectiveAutoencoder, self).__init__()
        # Initialize encoder layers
        self.enc1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.enc2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        
        # self.enc3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        
        
        self.enc4 = nn.Conv2d(32, 64, 3, stride=2, padding=1)

        self.fc1 = nn.Linear(64 * 32 * 32, 256)  
        self.fc2 = nn.Linear(256, 128)

        # Initialize decoder layers
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 64 * 32 * 32)
        self.dec1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
    
    
        # self.dec2 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        
        self.dec3 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.dec4 = nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = F.relu(self.enc1(x))  # layer 1
        x = F.relu(self.enc2(x))  
       
        # x_mid = F.relu(self.enc4(x))
        # print(x_mid.shape)
        
        x = x.view(x.size(0), -1)
        embedding = F.relu(self.fc2(F.relu(self.fc1(x)))) 
        
        x = F.relu(self.fc3(embedding))  
        x = F.relu(self.fc4(x)).view(-1, 64, 32, 32)
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec3(x))  
        # print(x.shape)
        output = torch.sigmoid(self.dec4(x))  
        return embedding, output


def ae_loss(input_img, output_img, ground_truth):
    diff = torch.abs(input_img - output_img)
    final = input_img - diff  
    return F.mse_loss(final, ground_truth)  
