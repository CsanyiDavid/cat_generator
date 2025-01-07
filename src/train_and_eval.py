import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

def kl_loss(mean, logvar):
    kl_loss = -0.5 * torch.sum(1 + logvar - torch.square(mean) - torch.exp(logvar))
    return kl_loss

def train(dl, model, reconstruction_loss, beta_fn, optimizer, epochs):
    model.train()
    dataset_size = len(dl.dataset)
    num_batches = len(dl)
    history = {
        "reconstruction_loss": [],
        "kl_loss": [],
        "vae_loss": [],
        "betas": [],
        "epoch_numbers" : [],
        "batches_per_epoch" : num_batches,
        "epochs": epochs 
    }

    for e in range(epochs):
        print(f"Epoch {e+1}")

        reconstruction_loss_values = []
        kl_loss_values = []
        vae_loss_values = []
        betas = []

        loop = tqdm(dl, desc="", unit="batch")
        for batch, (x, y) in enumerate(loop):
            pred, mean, logvar = model(x)

            reconstruction_loss_value = reconstruction_loss(pred, x)
            kl_loss_value = kl_loss(mean, logvar)
            beta = beta_fn(kl_loss_value.item()/model.latent_space_dim, e)
            vae_loss_value = reconstruction_loss_value + beta * kl_loss_value

            reconstruction_loss_values.append(reconstruction_loss_value.item())
            kl_loss_values.append(kl_loss_value.item())
            vae_loss_values.append(vae_loss_value.item())
            betas.append(beta)

            optimizer.zero_grad()
            vae_loss_value.backward()
            optimizer.step()

            if batch%5==0:
                loop.set_postfix(
                    reconstruction_loss=reconstruction_loss_value.item(),
                    kl_loss= kl_loss_value.item(),
                    beta=beta,
                    vae_loss = vae_loss_value.item()
                )

        mean_reconstruction_loss = sum(reconstruction_loss_values)/num_batches
        mean_kl_loss = sum(kl_loss_values)/num_batches
        mean_vae_loss = sum(vae_loss_values)/num_batches
        print(f"Train reconstruction loss: {mean_reconstruction_loss:>8f}")
        print(f"Train kl loss: {mean_kl_loss:>8f}")
        print(f"Train vae loss: {mean_vae_loss:>8f}\n")

        history["reconstruction_loss"].extend(reconstruction_loss_values)
        history["kl_loss"].extend(kl_loss_values)
        history["vae_loss"].extend(vae_loss_values)
        history["betas"].extend(betas)
        history["epoch_numbers"].extend([e]*num_batches)

    print("Done!")
    return history

def plot_loss_values(history):
    #Batch numbers when epochs ended
    epoch_ends = [history["batches_per_epoch"]*e for e in range(history["epochs"])]

    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    for (i, loss) in enumerate(["reconstruction_loss", "kl_loss", "vae_loss"]):
        axs[i].set_title(loss.replace('_', ' ').capitalize())
        axs[i].plot(history[loss])
        axs[i].set_xlabel("Batch")
        sec = axs[i].secondary_xaxis(-0.15)
        sec.set_xticks(epoch_ends, labels=range(history["epochs"]))
        sec.set_xlabel("Epoch")
    plt.show()
