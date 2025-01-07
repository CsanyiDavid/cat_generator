import sys
import os
src_path = os.path.join(os.getcwd(), '..')
sys.path.append(src_path)

import pickle
import torch
import torch.nn as nn

from src.models import VariationalAutoencoder
from src.train_and_eval import get_default_dataloader, train, plot_loss_values

def beta_fn(kl_loss, epoch):
    return 0.0001

def main():
    training_id = "tr0"
    dl = get_default_dataloader()
    vae = VariationalAutoencoder(
        input_shape=(1, 64, 64),
        latent_space_dim=128,
        encoder_conv_channels=(16, 32, 32, 64, 64),
        encoder_conv_kernel_size=(3, 3, 3, 3, 3),
        encoder_conv_stride=(2, 2, 2, 2, 2),
        encoder_conv_padding=(1, 1, 1, 1, 1),
        decoder_conv_t_channels=(64, 32, 16, 16, 1),
        decoder_conv_t_kernel_size=(3, 3, 3, 3, 3),
        decoder_conv_t_stride=(2, 2, 2, 2, 2),
        decoder_conv_t_padding=(1, 1, 1, 1, 1)
    )
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    epochs = 25

    history = train(dl, vae, mse, beta_fn, optimizer, epochs)
    with open(f'./../train_logs/{training_id}_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    plot_loss_values(history, f"./../train_logs/{training_id}_losses.png")
    torch.save(vae.state_dict(), f"./../train_logs/{training_id}_model.pth")

if __name__== "__main__":
    main()