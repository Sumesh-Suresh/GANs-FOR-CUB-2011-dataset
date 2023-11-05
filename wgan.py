import os
os.environ['PYTORCH_JIT'] = '1'

import torch
from utils import get_args
import torch.autograd as autograd

from networks import Discriminator, Generator
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    """
    Implementing WGAN-GP loss for discriminator.
    loss = E[D(fake_data)] - E[D(real_data)] + lambda * E[(|| grad wrt interpolated_data (D(interpolated_data))|| - 1)^2]
    """
   
    # WGAN-GP loss for discriminator.
    # loss_pt1 = E[D(fake_data)] - E[D(real_data)]
    # loss_pt2 = lambda * E[(|| grad wrt interpolated_data (D(interpolated_data))|| - 1)^2]
    # loss = loss_pt1 + loss_pt2
  
    loss_pt1 = torch.mean(discrim_fake) - torch.mean(discrim_real)
    gradients = torch.autograd.grad(outputs=discrim_interp, 
                                    inputs=interp, 
                                    grad_outputs=torch.ones(discrim_interp.size()).to(interp.device),
                                    create_graph=True, 
                                    # retain_graph=True, 
                                    # only_inputs=True
                                )[0]
    loss_pt2 = lamb * torch.mean((torch.linalg.norm(gradients, dim=1, ord=2) - 1) ** 2)
    loss = loss_pt1 + loss_pt2

    return loss


def compute_generator_loss(discrim_fake):
   
    # Implement WGAN-GP loss for generator.
    # loss = - E[D(fake_data)]
    
    loss = -1*torch.mean(discrim_fake)
    
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_wgan_gp/"
    os.makedirs(prefix, exist_ok=True)

    train_model(
        gen,
        disc,
        num_iterations=int(5e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
