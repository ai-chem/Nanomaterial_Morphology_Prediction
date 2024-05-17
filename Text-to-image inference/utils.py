import pandas as pd
import random
import glob
from transformers import AutoTokenizer, AutoModel
import torch
import pytorch_lightning as pl
from matplotlib import pyplot
import os
import numpy as np
import random
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import urllib.parse
from torch.nn import functional as F
from pl_bolts import _HTTPS_AWS_HUB
from pytorch_lightning import LightningModule, Trainer, seed_everything
import torch
import matplotlib as plt
import pandas as pd
import pickle
from matplotlib import pyplot
import os
import sys
from Linking_AE_training.VAE_Final_Architecture import VAE

def generate_patterns(procedures, values, index = 0, num_of_patterns = 1):  
  new_procedures = []
  for i in range(num_of_patterns):
    pattern = random.choice(procedures)
    ca_conc = values.iloc[index, 0]
    co3_conc = values.iloc[index, 1]
    hco3_conc = values.iloc[index, 2]
    polymer = values.iloc[index, 3]
    pol_mass = values.iloc[index, 4]
    pol_conc = values.iloc[index, 5]
    pol_vol = 20
    surfactant = values.iloc[index, 6]
    surf_conc = values.iloc[index, 7]
    surf_vol = 20
    solvent = values.iloc[index, 8]
    solvent_volume = values.iloc[index, 9]
    stir_ratio = values.iloc[index, 10]
    r_time = values.iloc[index, 12]
    r_temp = values.iloc[index, 11]
    new_procedure = pattern.format(ca_conc=ca_conc,
                                      co3_conc=co3_conc,
                                      hco3_conc=hco3_conc, 
                                      polymer=polymer,
                                      pol_mass=pol_mass,
                                      pol_conc=pol_conc,
                                      pol_vol=pol_vol,
                                      surfactant=surfactant,
                                      surf_conc=surf_conc,
                                      surf_vol=surf_vol,
                                      solvent=solvent, 
                                      solvent_volume=solvent_volume,
                                      stir_ratio=stir_ratio,
                                      r_time=r_time,
                                      r_temp=r_temp)
    new_procedures = new_procedures + [new_procedure] 
  
  return new_procedures

def Create_data(path, pattern = None, db_name = "synthesis_conditions_dataset.xlsx"):
    path = path + 'Version_0'
    with open('Datasets\procedure_patterns.txt') as f:
        lines = filter(None, (line.rstrip() for line in f))
        procedures = list(lines)
    if pattern != None:
       procedures = [procedures[pattern]]
    values = pd.read_excel('Datasets\{}'.format(db_name))
    df = pd.DataFrame(columns = ['image', 'text'])
    df['image'] = pd.Series([file.replace(path + '\\','') for file in glob.glob(path + '\\*.jpg')])
    for index, row in enumerate(df.image):
        try:
          df.iloc[index, 1] = generate_patterns(procedures, values, int([x for x in row.split('_') if x.isdigit()][0]))[0]
        except:
           pass
    return df.dropna()

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def Load_model(device):
    #Загрузка токенизатора и самой модели
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").to(device)
    return model, tokenizer

def Create_text_embeddings(preprocessed_data, device, batch_size = 200):
    model, tokenizer = Load_model(device)
    bert_input = list(chunks(preprocessed_data.text.to_list(), batch_size))
    embeddings = torch.tensor([])
    for index, batch in enumerate(bert_input):
        tokenized_train = tokenizer(batch, padding = 'max_length', truncation = True, return_tensors="pt", max_length=256)
        #move on device (GPU)
        tokenized_train = {k:v.clone().detach().to(device) for k,v in tokenized_train.items()}
        with torch.no_grad():
            hidden_train = model(**tokenized_train) #dim : [batch_size(nr_sentences), tokens, emb_dim]
            embeddings = torch.cat((embeddings, hidden_train.last_hidden_state[:, :, :,].detach().cpu()[:,0]), 0)
        print('Batch', str(index + 1), 'completed out of', len(bert_input))
    return embeddings

def create_dataset(path_to_data, batch_size=32, crop_size=128, num_of_channels=1):
    transform = transforms.Compose(
        [
            #transforms.RandomRotation(degrees=(0, 360)),
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=num_of_channels),
        ]
    )
    dataset = ImageFolder(root=path_to_data, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=8,
    )
    return dataset, dataloader

def Create_image_embeddings(path_to_data, ckpt_path, batch_size = 156, crop_size = 128, num_of_channels = 1):
    model = VAE.load_from_checkpoint(ckpt_path)
    model.eval()
    dataset, dataloader = create_dataset(
        path_to_data, batch_size, crop_size, num_of_channels
    )
    all_z = torch.tensor([])
    for batch in dataloader:
        input_imgs = batch[0].to(model.device)
        with torch.no_grad():
            model.eval()
            x = model.encoder(input_imgs)
            mu = model.fc_mu(x)
            log_var = model.fc_var(x)
            p, q, z = model.sample(mu, log_var)
            model.train()
        all_z = torch.cat((all_z, z), 0)
        print(all_z.shape[0], "images preprocessed out of", len(dataset))
    return all_z

def Reconstruct(embedding, ckpt_path):
    model = VAE.load_from_checkpoint(ckpt_path)
    decoded = model.decoder(embedding)
    return decoded     

def Plot_results(reconstruct, real, initial, save_to, ckpt_path):
    for i in range(len(real)):
        pyplot.figure(figsize=(20,12))
        pyplot.subplot(131)
        pyplot.imshow(initial[i][0], cmap="gray")
        pyplot.title("Initial")
        pyplot.subplot(132)
        pyplot.imshow(Reconstruct(real, ckpt_path)[i].cpu().detach().numpy()[0], cmap="gray")
        pyplot.title("Reconstructed")
        pyplot.subplot(133)
        pyplot.imshow(Reconstruct(reconstruct, ckpt_path)[i].cpu().detach().numpy()[0], cmap="gray")
        pyplot.title("Generated")
        if save_to is not None:
            if not os.path.exists(save_to + "recs/"):
                os.makedirs(save_to + "recs/")
            pyplot.savefig(save_to + "recs/{}.pdf".format(i))
        else:
            pyplot.show()
    pyplot.close("all")