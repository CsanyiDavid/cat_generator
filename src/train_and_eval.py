import torch
import torch.nn as nn
import torchvision as tv
from torchvision.transforms import v2
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.image_processing import RandomCropResizeTransform

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

def plot_loss_values(history, figname=None):
    #Batch numbers when epochs ended
    epoch_ends = [history["batches_per_epoch"]*e for e in range(history["epochs"])]

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    for (i, loss) in enumerate(["reconstruction_loss", "kl_loss", "vae_loss"]):
        axs[i].set_title(loss.replace('_', ' ').capitalize())
        axs[i].plot(history[loss])
        axs[i].set_xlabel("Batch")
        sec = axs[i].secondary_xaxis(-0.15)
        sec.set_xticks(epoch_ends, labels=range(history["epochs"]))
        sec.set_xlabel("Epoch")
    fig.tight_layout()
    if figname is None:
        plt.show()
    else:
        plt.savefig(figname)

def get_default_dataloader():
    transforms = [
        tv.transforms.Grayscale(),
        tv.transforms.ToTensor(),
        v2.RandomHorizontalFlip(0.5),
        v2.ColorJitter(0.2, 0.2, 0.2, 0.2),
        RandomCropResizeTransform(38, 64, 0.5)
    ]
    ds1 = tv.datasets.ImageFolder(
        root="./../data/cats1",
        transform=tv.transforms.Compose(transforms)
    )
    ds2 = tv.datasets.ImageFolder(
        root="./../data/cats2/data",
        transform=tv.transforms.Compose(transforms)
    )
    ds = torch.utils.data.ConcatDataset([ds1, ds2])
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=32,
        shuffle=True
    )
    return dl