import argparse
import os
os.environ['PYTORCH_JIT'] = '1'
device ='cuda'
from utils import get_args

import torch

from networks import Discriminator, Generator
import torch.nn.functional as F
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
   
    criterion = torch.nn.functional.binary_cross_entropy_with_logits
    
    real_labels = torch.ones_like(discrim_real)
    fake_labels = torch.zeros_like(discrim_fake)
    loss_real = criterion(discrim_real, real_labels)
    loss_fake = criterion(discrim_fake, fake_labels)
    loss = loss_real + loss_fake
   
    return loss


def compute_generator_loss(discrim_fake):
   
    criterion = torch.nn.functional.binary_cross_entropy_with_logits
    real_labels = torch.ones_like(discrim_fake)  
    loss = criterion(discrim_fake, real_labels)
    return loss
   

if __name__ == "__main__":
    args = get_args()
    gen = Generator().to(device)
    disc = Discriminator().to(device)
   
    prefix = "data_gan/"
    os.makedirs(prefix, exist_ok=True)

    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
