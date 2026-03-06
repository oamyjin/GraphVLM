import torch
import wandb
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader
import re

pred = 'cs.CV(Computer Vision and Pattern Recognition)'
label = 'cs.CV(Computer Vision and Pattern Recognitionaaa)'

clean_pred = re.sub(r'\(.*\)', '', pred.strip()).strip()
clean_label = re.sub(r'\(.*\)', '', label.strip()).strip()
print(clean_label)
matches = re.findall(r"cs\.[a-zA-Z]{2}", clean_pred)

correct = 0
if len(matches) > 0 and clean_label == matches[0]:
    correct += 1
    print('correct')
    print(f'gt: {clean_label}')