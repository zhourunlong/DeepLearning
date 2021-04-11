import argparse
import torch
import torch.optim as optim
import numpy as np
from torch import nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import os, time


class CVAE(nn.Module):
    def __init__(self, img_size, label_size, latent_size, hidden_size=256):
        super(CVAE, self).__init__()
        self.img_size = img_size  # (C, H, W)
        self.label_size = label_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        # Encoder.
        '''
        img   -> fc  ->                   -> fc -> mean    
                        concat -> encoder                  -> z
        label -> fc  ->                   -> fc -> logstd 
        '''
        self.enc_img_fc = nn.Linear(int(np.prod(self.img_size)), self.hidden_size)
        self.enc_label_fc = nn.Linear(self.label_size, self.hidden_size)
        self.encoder = nn.Sequential(
            nn.Linear(2 * self.hidden_size, 2 * self.hidden_size), nn.ReLU(),
            nn.Linear(2 * self.hidden_size, 2 * self.hidden_size), nn.ReLU(),
        )
        self.z_mean = nn.Linear(2 * self.hidden_size, self.latent_size)
        self.z_logstd = nn.Linear(2 * self.hidden_size, self.latent_size)
        # Decoder.
        '''
        latent -> fc ->
                         concat -> decoder -> reconstruction
        label  -> fc ->
        '''
        self.dec_latent_fc = nn.Linear(self.latent_size, self.hidden_size)
        self.dec_label_fc = nn.Linear(self.label_size, self.hidden_size)
        self.decoder = nn.Sequential(
            nn.Linear(2 * self.hidden_size, 2 * self.hidden_size), nn.ReLU(),
            nn.Linear(2 * self.hidden_size, 2 * self.hidden_size), nn.ReLU(),
            nn.Linear(2 * self.hidden_size, int(np.prod(self.img_size))), nn.Sigmoid(),
        )
        # TODO: assume the distribution of reconstructed images is a Gaussian distibution. Write the log_std here.
        self.recon_logstd = None

    def encode(self, batch_img, batch_label):
        '''
        :param batch_img: a tensor of shape (batch_size, C, H, W)
        :param batch_label: a tensor of shape (batch_size, self.label_size)
        :return: a batch of latent code of shape (batch_size, self.latent_size)
        '''
        # TODO: compute latent z from images and labels
        return None  # Placeholder.

    def decode(self, batch_latent, batch_label):
        '''
        :param batch_latent: a tensor of shape (batch_size, self.latent_size)
        :param batch_label: a tensor of shape (batch_size, self.label_size)
        :return: reconstructed results
        '''
        return None  # Placeholder.

    def sample(self, batch_latent, batch_label):
        '''
        :param batch_latent: a tensor of size (batch_size, self.latent_size)
        :param batch_label: a tensor of size (batch_size, self.label_dim)
        :return: a tensor of size (batch_size, C, H, W), each value is in range [0, 1]
        '''
        with torch.no_grad():
            # TODO: get samples from the decoder.
            pass
        return None  # Placeholder.


#########################
####  DO NOT MODIFY  ####
def generate_samples(cvae, n_samples_per_class, device):
    cvae.eval()
    latent = torch.randn((n_samples_per_class * 10, cvae.latent_size), device=device)
    label = torch.eye(cvae.label_size, dtype=torch.float, device=device).repeat(n_samples_per_class, 1)
    imgs = cvae.sample(latent, label).cpu()
    label = torch.argmax(label, dim=-1).cpu()
    samples = dict(imgs=imgs, labels=label)
    return samples
#########################


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    # Load dataset
    if args.dataset == "mnist":
        dataset = MNIST(root="../data",
                        transform=transforms.ToTensor(),  # TODO: you may want to tweak this
                        train=not args.eval)
        dataloader = DataLoader(dataset, args.batch_size, shuffle=True, drop_last=True)
    else:
        raise NotImplementedError

    # Configure
    logdir = args.logdir if args.logdir is not None else "/tmp/cvae_" + time.strftime('%Y-%m-%d-%H-%M-%S')
    os.makedirs(logdir, exist_ok=True)

    label_dim = 10
    img_dim = (1, 28, 28)
    latent_dim = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cvae = CVAE(img_dim, label_dim, latent_dim)
    cvae.to(device)
    optimizer = optim.Adam(cvae.parameters(), lr=args.lr)

    if not args.eval:
        for name, param in cvae.named_parameters():
            print(name, param.shape)
        prior = torch.distributions.Normal(0, 1)
        for epoch in range(args.num_epochs):
            # TODO: Training, logging, saving, visualization, etc.
            pass
    else:
        assert args.load_path is not None
        checkpoint = torch.load(args.load_path, map_location=device)
        cvae.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        cvae.eval()
        samples = generate_samples(cvae, 1000, device)
        torch.save(samples, "vae_generated_samples.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", choices=["mnist"], default="mnist")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--load_path", type=str, default=None)
    args = parser.parse_args()
    main(args)
