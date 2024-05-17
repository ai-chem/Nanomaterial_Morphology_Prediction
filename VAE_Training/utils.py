import torch
import random
import numpy
from PIL import Image
import os
from VAE_Architectures import VAE
import pandas as pd
from piqa import PSNR, SSIM
import torchvision.transforms as transforms
from matplotlib import pyplot
from matplotlib.lines import Line2D
import seaborn as sns
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def create_dataset(path_to_data, batch_size=32, crop_size=128, num_of_channels=1):
    transform = transforms.Compose(
        [
            transforms.RandomRotation(degrees=(0, 360)),
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=num_of_channels),
        ]
    )
    dataset = ImageFolder(root=path_to_data, transform=transform)
    g = torch.Generator()
    g.manual_seed(0)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        worker_init_fn=seed_worker,
        generator=g,
    )
    return dataset, dataloader

def create_dataset_train_test_split(path_to_data, batch_size=32, crop_size=128, num_of_channels=1):
    transform = transforms.Compose(
        [
            transforms.RandomRotation(degrees=(0, 360)),
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=num_of_channels),
        ]
    )
    dataset = ImageFolder(root=path_to_data, transform=transform)
    dataset_train, dataset_test = train_test_split(dataset, test_size=0.1, random_state=41)

    dataset_test, dataset_val = train_test_split(dataset_test, test_size=0.5, random_state=41)
    
    g = torch.Generator()
    g.manual_seed(0)

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        worker_init_fn=seed_worker,
        generator=g,
    )

    dataloader_test = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        worker_init_fn=seed_worker,
        generator=g,
    )
    
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        worker_init_fn=seed_worker,
        generator=g,
    )
    return dataset_train, dataset_test, dataset_val, dataloader_train, dataloader_test, dataloader_val

def get_train_images(number_of_images, dataset, rand=True):
    if rand == True:
        return torch.stack(
            [dataset[i][0] for i in random.sample(range(0, len(dataset)), number_of_images)], dim=0
        )
    else:
        return torch.stack([dataset[i][0] for i in range(number_of_images)], dim=0)

def get_metrics(path):
    im1 = Image.open(path + "tmp1.png")
    im2 = Image.open(path + "tmp2.png")

    transform = transforms.Compose([transforms.Grayscale(3), transforms.ToTensor()])

    x = transform(im1).unsqueeze(0).cuda()
    y = transform(im2).unsqueeze(0).cuda()

    psnr = PSNR()
    ssim = SSIM().cuda()
    return psnr(x, y), ssim(x, y)

def plot_vae_images(input_images, reconstructed_images, save_path, save_images = True):
    psnr_metrics = []
    ssim_metrics = []
    for i in range(len(input_images)):
        current_input_image = input_images[i].cpu().detach().numpy()[0]
        current_reconstructed_image = reconstructed_images[i].cpu().detach().numpy()[0]
        pyplot.figure()
        pyplot.axis("off")
        pyplot.imshow(current_input_image, cmap="gray")
        pyplot.savefig(
            save_path + "tmp1.png", bbox_inches="tight", pad_inches=0
        )
        pyplot.figure()
        pyplot.axis("off")
        pyplot.imshow(current_reconstructed_image, cmap="gray")
        pyplot.savefig(
            save_path + "tmp2.png", bbox_inches="tight", pad_inches=0
        )
        pyplot.close("all")
        psnr, ssim = get_metrics(save_path)
        psnr_metrics.append(float(psnr))
        ssim_metrics.append(float(ssim))
        if save_images == True:
            pyplot.figure()
            pyplot.subplot(121)
            pyplot.imshow(current_input_image, cmap="gray")
            pyplot.title("Original")
            pyplot.subplot(122)
            pyplot.imshow(current_reconstructed_image, cmap="gray")
            pyplot.title("Reconstruction")
            pyplot.savefig(save_path + "{}.pdf".format(i))
            pyplot.close("all")
    return psnr_metrics, ssim_metrics

def create_loss_plot(log_dir, save_path):
    event_accumulator = EventAccumulator(log_dir)
    event_accumulator.Reload()
    df_metrics = pd.read_csv(save_path + 'df_metrics.csv')

    events = event_accumulator.Scalars("train_recon_loss")
    x = [x.step for x in events]
    y = [x.value for x in events]

    df_loss = pd.DataFrame({"Step": x, "Loss": y}).drop(columns=['Step'])
    pyplot.figure(figsize=(10, 4))
    pyplot.subplot(121)  
    pyplot.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4) 
    g = sns.lineplot(data=df_metrics.PSNR, color='#c698b7')
    pyplot.xlabel("Epoch")
    sns.lineplot(data=df_metrics.SSIM, color="#f99779", ax=g.axes.twinx())

    g.legend(handles=[Line2D([], [], marker='_', color="#c698b7", label='PSNR'), Line2D([], [], marker='_', color="#f99779", label='SSIM')])
    pyplot.title('Metrcis by epoch')

    pyplot.subplot(122) 
    g = sns.lineplot(data=df_loss, color='#c698b7')
    pyplot.xlabel("Epoch")
    pyplot.ylabel("Loss")
    pyplot.title('Loss by epoch')
    pyplot.savefig(save_path + "loss_plot.png")

def validation(dataset_val, checkpoint_path, save_path, images_number):
    save_path = save_path +"Validation_results/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    input_images = get_train_images(images_number, dataset_val, rand=True)
    model = VAE.load_from_checkpoint(checkpoint_path)
    with torch.no_grad():
        model.eval()
        reconstructed_images = model(input_images)
        model.train()
    psnr_metrics, ssim_metrics = plot_vae_images(input_images, reconstructed_images, save_path, save_images = False)
    return psnr_metrics, ssim_metrics