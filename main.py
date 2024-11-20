
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from utils.utils import Interpolate, gridImage
from model_gc import GCNet
from utils.dataloader import gtTestImageDataset, testImageDataset, mean, std, padsize

def convert_to_numpy(input,H,W):
    image = input[:,:,padsize:H-padsize,padsize:W-padsize].clone()
    input_numpy = image[:,:,:H,:W].clone().cpu().numpy().reshape(3,H-padsize*2,W-padsize*2).transpose(1,2,0)
    for i in range(3):
        input_numpy[:,:,i] = input_numpy[:,:,i] * std[i] + mean[i]

    return  input_numpy




def convert_to_numpy(input,H,W):
    image = input[:,:,padsize:H-padsize,padsize:W-padsize].clone()
    input_numpy = image[:,:,:H,:W].clone().cpu().numpy().reshape(3,H-padsize*2,W-padsize*2).transpose(1,2,0)
    for i in range(3):
        input_numpy[:,:,i] = input_numpy[:,:,i] * std[i] + mean[i]

    return  input_numpy

dataset = "test_dataset"

# make output directory
os.makedirs("../.dataset/" + dataset + "/output", exist_ok=True)

if torch.cuda.is_available(): device , torch.backends.cudnn.benchmark = 'cuda' , True
else:device = 'cpu'

# Initialize generator
Generator = GCNet().to(device)
Generator.eval()
Generator.load_state_dict(torch.load("../checkpoint/model_gc.pth", map_location=device,weights_only=True))

# read image
gtAvailable = False
if os.path.exists("../.dataset/" + dataset + "/gt"):
    if len(os.listdir("../.dataset/" + dataset + "/input")) == len(os.listdir("../.dataset/" + dataset + "/gt")):
        gtAvailable = True

if gtAvailable:
    image_dataset = gtTestImageDataset("../.dataset/" + dataset)
else:
    image_dataset = testImageDataset("../.dataset/" + dataset)

# run
all_psnr = 0.0
all_ssim = 0.0
print("[Dataset name: %s] --> %d images" % (dataset, len(image_dataset)))
for image_num in tqdm(range(len(image_dataset)),ncols=100):

    data = image_dataset[image_num]
    R = data["R"].to(device)

    _,first_h,first_w = R.size()
    R = torch.nn.functional.pad(R,(0,(R.size(2)//16)*16+16-R.size(2),0,(R.size(1)//16)*16+16-R.size(1)),"constant")
    R = R.view(1,3,R.size(1),R.size(2))
    with torch.no_grad():output  = Generator(R) 

    #output image
    output_np = np.clip(convert_to_numpy(output,first_h,first_w) + 0.015,0,1)
    R_np = convert_to_numpy(R,first_h,first_w)
    final_output = np.fmin(output_np, R_np)

    # save output
    Image.fromarray(np.uint8(final_output * 255)).save("../.dataset/" + dataset + "/output/" + data["Name"] + ".png")

    # Calculate PSNR/SSIM if available
    if gtAvailable:
        B = data["B"].astype(np.float32)
        # print(B.shape,final_output.shape)
        thisPSNR = psnr(B, final_output.astype(np.float32))
        thisSSIM = ssim(B, final_output.astype(np.float32), multichannel=True, channel_axis=2,data_range=255.0)
        all_psnr += thisPSNR
        all_ssim += thisSSIM
        print("[%s] PSNR:%4.2f SSIM:%4.3f" % (data["Name"], thisPSNR, thisSSIM), end="\r")

if gtAvailable:
    all_psnr = all_psnr/len(image_dataset)
    all_ssim = all_ssim/len(image_dataset)
    print("hogya : [[%s]]" % (dataset))
    print("PSNR: %4.2f / SSIM: %4.3f" % (all_psnr, all_ssim))
else:
    print("Complete.")