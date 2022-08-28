from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import ImageFont, ImageDraw, Image
from fontTools import ttLib
import numpy as np
import torch
from torch import nn
import pandas as pd
import glob 


ROOTDIR = "C:/Users/liury/OneDrive/桌面/ZiNet/ZiNetDataset"
FONTDIR = f"{ROOTDIR}/fonts"
PATHS = [path[len(FONTDIR):] for path in glob.glob(f'{FONTDIR}/*')] # list of font names
fonts = [ttLib.TTFont(FONTDIR + path) for path in PATHS]
chars = pd.read_csv(f"{ROOTDIR}/edu_standard.csv")

class ZiCDataset(Dataset):

  def __init__(self, hparams, stage = "train", device = "cpu"):
    """
    hparams: 
      fontdir:     font directory,
      supports:    font support array,
      train_split: batches per split; train/val/test,
      image_size:  image size per character,
      font_size:   font size in the image,
      device:      device the model is on
    """
    super().__init__()

    self.LEN = 2 ** 15
    self.hparams = hparams
    self.stage = stage
    self.device = device
    
    self.supports = np.load("C:/Users/liury/OneDrive/桌面/ZiNet/ZiNetDataset/font_support.npy")

    assert(len(self.hparams['train_split']) == 3) #train/val/test

    self.train_split_size = sum(self.hparams['train_split'])
    self.train_split_cumsum = np.cumsum(self.hparams['train_split'])


    self.fontcnt = self.supports.shape[0] #how many fonts
    self.charcnt = self.supports.shape[1] #how many chars

  def __len__(self):
    if(self.stage == "train"):
      return self.hparams["batch_size"] * self.hparams['train_split'][0]
    elif(self.stage == "val"):
      return self.hparams["batch_size"] * self.hparams['train_split'][1]
    elif(self.stage == "test"):
      return self.hparams["batch_size"] * self.hparams['train_split'][2]
    else:
      raise ValueError('Invalid stage specified')
  
   


  def getChar(self, font_idx, chr_idx):
      """
      @param font_idx: int, index of font
      @param chr_idx: int, index of character
      @return Tensor: W * W bitmap of font (-1 ~ 1)
      """
 
      chr = chars["Character"][chr_idx]
      font_dir = f'{"C:/Users/liury/OneDrive/桌面/ZiNet/ZiNetDataset/fonts"}{PATHS[font_idx]}'
      W = self.hparams["image_size"]
    
      image = Image.new(mode = "L", size = (W, W), color = 0)
      font = ImageFont.truetype(font_dir, self.hparams["font_size"])
      mask = font.getmask(chr)
      w, h = mask.size
      d = Image.core.draw(image.im, 0)
      d.draw_bitmap(((W - w)/2, (W - h)/2), mask, 255) # last arg is pixel intensity of text  
      #plt.imshow(image)

      

      transform = transforms.Compose([
          transforms.PILToTensor()
      ])
    
      # transform = transforms.PILToTensor()
      # Convert the PIL image to Torch tensor
      tensor_form = transform(image)
      range = tensor_form.max().item() - tensor_form.min().item()
      minval = tensor_form.min().item()


      return (tensor_form - minval) / range * 2 - 1

  def __getitem__(self, idx):

    
    fontidx = np.random.randint(self.fontcnt)
    while True:
      idx1 = np.random.randint(self.charcnt)
      while not self.supports[fontidx][idx1]:
        idx1 = np.random.randint(self.charcnt)
      
      A = (int)(1e9 + 7)
      # (A * idx1 + idx2) = required 
      offset = 0
      len    = 0
      # idx2 % self.train_split_size is random(0 ~ len) + offset
      if (self.stage == "train"):
        offset = 0
        len    = self.hparams['train_split'][0]
      elif (self.stage == "val"):
        offset = self.train_split_cumsum[0]
        len    = self.hparams['train_split'][1]
      elif (self.stage == "test"):
        offset = self.train_split_cumsum[1]
        len    = self.hparams['train_split'][2]
      
      idx2_modulo   = (np.random.randint(len) + offset) % self.train_split_size
      idx2_quotient = np.random.randint((self.charcnt - idx2_modulo) // self.train_split_size)
      
      idx2 = idx2_quotient * self.train_split_size + idx2_modulo

      while not self.supports[fontidx][idx2]:
        
        idx2_modulo   = (np.random.randint(len) + offset) % self.train_split_size
        idx2_quotient = np.random.randint((self.charcnt - idx2_modulo) // self.train_split_size)
        idx2 = idx2_quotient * self.train_split_size + idx2_modulo

      image1 = self.getChar(fontidx, idx1)
      image2 = self.getChar(fontidx, idx2)

      if(image1.min().item() == image1.max().item() or image2.min().item() == image2.max().item() or torch.isnan(image1).any() or torch.isnan(image2).any()):
        continue

      return {
          "in_imgs" : image1,
          "in_idxs" : torch.LongTensor([idx1], device = self.device),
          "out_imgs": image2,
          "out_idxs": torch.LongTensor([idx2], device = self.device),
          "font":     torch.LongTensor([fontidx], device = self.device),
      }

def make_mlp(
    input_size,
    hidden_size,
    output_size,
    hidden_layers,
    hidden_activation="GELU",
    output_activation="GELU",
    layer_norm=False,
):
    """Construct an MLP with specified fully-connected layers."""
    hidden_activation = getattr(nn, hidden_activation)
    if output_activation is not None:
        output_activation = getattr(nn, output_activation)
    layers = []
    sizes = [input_size] + [hidden_size]*(hidden_layers-1) + [output_size]
    # Hidden layers
    for i in range(hidden_layers-1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i + 1]))
        layers.append(hidden_activation())
    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1]))
        layers.append(output_activation())
    return nn.Sequential(*layers)