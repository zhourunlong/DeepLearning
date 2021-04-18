import argparse, sys, os
import torch, torchvision
import torch.optim as optim
import numpy as np
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
import os, time, logging
from tqdm import tqdm
from mnist_classifier import MnistClassifier
import matplotlib.pyplot as plt

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

        input = torch.cat((self.enc_img_fc(batch_img.view(batch_img.shape[0], -1)), self.enc_label_fc(batch_label)), dim = 1)
        output = self.encoder(input)
        mu = self.z_mean(output)
        logsigma = self.z_logstd(output)

        dz = torch.randn((batch_img.shape[0], self.latent_size), device=batch_img.device)
        z = mu + torch.exp(0.5 * logsigma) * dz

        return z, mu, logsigma

    def decode(self, batch_latent, batch_label):
        '''
        :param batch_latent: a tensor of shape (batch_size, self.latent_size)
        :param batch_label: a tensor of shape (batch_size, self.label_size)
        :return: reconstructed results
        '''
        input = torch.cat((self.dec_latent_fc(batch_latent), self.dec_label_fc(batch_label)), dim = 1)
        output = self.decoder(input)

        return output

    def forward(self, batch_img, batch_label):
        z, mu, logsigma = self.encode(batch_img, batch_label)
        x = self.decode(z, batch_label)
        return x, mu, logsigma

    def sample(self, batch_latent, batch_label):
        '''
        :param batch_latent: a tensor of size (batch_size, self.latent_size)
        :param batch_label: a tensor of size (batch_size, self.label_dim)
        :return: a tensor of size (batch_size, C, H, W), each value is in range [0, 1]
        '''
        with torch.no_grad():
            input = torch.cat((self.dec_latent_fc(batch_latent), self.dec_label_fc(batch_label)), dim = 1)
            output = self.decoder(input)

            return output.clamp(0, 1).view(-1, 1, 28, 28)


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

def criterion(recon_x, x, mu, logsigma):
    Lrc = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction = "sum")
    DKL = 0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
    return Lrc - DKL

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    # Load dataset
    if args.dataset == "mnist":
        dataset = MNIST(root="../data",
                        transform=transforms.ToTensor(),
                        train=not args.eval)
        dataloader = DataLoader(dataset, args.batch_size, shuffle=True, drop_last=True)
    else:
        raise NotImplementedError

    # Configure
    logdir = args.logdir

    label_dim = 10
    img_dim = (1, 28, 28)
    latent_dim = 100

    device = torch.device("cuda:" + str(args.gpuid) if torch.cuda.is_available() else "cpu")
    cvae = CVAE(img_dim, label_dim, latent_dim)
    cvae.to(device)
    optimizer = optim.Adam(cvae.parameters(), lr=args.lr)

    validator = torch.load("mnist_classifier.pth", map_location=device)

    if not args.eval:
        for name, param in cvae.named_parameters():
            print(name, param.shape)
        prior = torch.distributions.Normal(0, 1)
        best_acc = 0
        
        for epoch in range(args.num_epochs):
            cvae.train()
            train_loss = 0
            idx = 0

            pbar = tqdm(total=len(dataloader.dataset))
            for batch_idx, (data, label) in enumerate(dataloader):
                idx = batch_idx
                data = data.to(device)
                label = F.one_hot(label.to(device), 10).float()
                optimizer.zero_grad()

                x, mu, logsigma = cvae(data, label)
                
                loss = criterion(x, data, mu, logsigma)
                train_loss += loss.item()

                loss.backward()
                optimizer.step()

                pbar.update(data.size(0))
                pbar.set_description('Train, loss: {:.6f}'.format(train_loss / (batch_idx + 1)))
                pbar.set_postfix(lr=args.lr)

            pbar.close()

            logging.info('Epoch {}. Train, loss: {:.6f}'.format(epoch, train_loss / (idx + 1)))

            save_path = os.path.join(args.logdir, "models/model{}.pth".format(epoch))
            torch.save(cvae, save_path)
            logging.info('Saving to {}'.format(save_path))

            img_queue = generate_samples(cvae, 10, device)['imgs']
            img_grid = torchvision.utils.make_grid(img_queue.cpu(), nrow = 10).numpy()
            plt.imshow(np.transpose(img_grid, (1,2,0)))
            save_path = os.path.join(args.logdir, "images/valid_batch{}.jpg".format(epoch))
            plt.savefig(save_path)

            num_per_class = 1024
            valid_data_queue = generate_samples(cvae, num_per_class, device)
            valid_data = valid_data_queue['imgs'].to(device)
            valid_label = valid_data_queue['labels'].to(device)

            acc = 0

            pbar = tqdm(total = num_per_class // 128 * 10)
            for i in range(num_per_class // 128 * 10):
                data = valid_data[i * 128: (i + 1) * 128, :, :, :]
                data = transforms.Normalize((0.1307,), (0.3081,))(data)
                label = F.one_hot(valid_label[i * 128: (i + 1) * 128], 10).float()
                
                with torch.no_grad():
                    output = validator(data)
                    pred = output.argmax(dim=1, keepdim = True)
                    acc += pred.eq(valid_label[i * 128: (i + 1) * 128].view_as(pred)).sum().item()

                pbar.update(1)
                pbar.set_description('Valid, acc: {:.6f}'.format(acc / (128 * (i+ 1))))
                pbar.set_postfix(lr=args.lr)
            
            pbar.close()

            acc /= num_per_class * 10
            logging.info('Valid, acc: {:.6f}'.format(acc))

            if acc > best_acc:
                best_acc = acc
                save_path = os.path.join(args.logdir, "models/best.pth")
                torch.save(cvae, save_path)
                logging.info('Best acc! Saving to {}'.format(save_path))

    else:
        assert args.load_path is not None
        cvae = torch.load(args.load_path, map_location=device)
        samples = generate_samples(cvae, 1000, device)
        torch.save(samples, "vae_generated_samples.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, default=2018011309)
    parser.add_argument("--dataset", choices=["mnist"], default="mnist")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--gpuid", type=int, default=0)
    args = parser.parse_args()

    if args.logdir is None:
        args.logdir = "Models-{}".format(time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(os.path.join(args.logdir, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.logdir, "images"), exist_ok=True)
    print("Experiment dir : {}".format(args.logdir))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.logdir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    main(args)
