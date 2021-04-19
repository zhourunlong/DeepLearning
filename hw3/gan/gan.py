import argparse, sys, os, logging
import torch, torchvision
import torch.optim as optim
import numpy as np
from torch import nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib
import matplotlib.pyplot as plt
from torch.nn import functional as F
import time
from mnist_classifier import MnistClassifier
from tqdm import tqdm

matplotlib.use('Agg')

std_threshold = torch.Tensor([0.17, 0.08, 0.17, 0.15, 0.14, 0.16, 0.15, 0.13, 0.15, 0.13])

class Generator(nn.Module):
    def __init__(self, latent_dim, label_dim, img_dim: tuple, hidden_size=256):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.label_dim = label_dim
        self.img_dim = img_dim  # (C, H, W)
        self.hidden_size = hidden_size
        # Layers
        self.latent_fc = nn.Linear(self.latent_dim, self.hidden_size)
        self.latent_bn = nn.BatchNorm1d(self.hidden_size)
        self.label_fc = nn.Linear(self.label_dim, self.hidden_size)
        self.label_bn = nn.BatchNorm1d(self.hidden_size)
        self.fc1 = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)
        self.bn1 = nn.BatchNorm1d(2 * self.hidden_size)
        self.fc2 = nn.Linear(2 * self.hidden_size, 4 * self.hidden_size)
        self.bn2 = nn.BatchNorm1d(4 * self.hidden_size)
        self.fc3 = nn.Linear(4 * self.hidden_size, int(np.prod(self.img_dim)))

        self._initialize(0., 0.02)

    def _initialize(self, mean, std):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()

    def forward(self, z, label):
        z = nn.functional.relu(self.latent_bn(self.latent_fc(z)))
        label = nn.functional.relu(self.label_bn(self.label_fc(label)))
        latent = torch.cat([z, label], dim=-1)
        out = nn.functional.relu(self.bn1(self.fc1(latent)))
        out = nn.functional.relu(self.bn2(self.fc2(out)))
        out = torch.tanh(self.fc3(out))
        out = torch.reshape(out, (-1,) + self.img_dim)
        return out

    def sample(self, z, label):
        '''
        :param z: latent z with size (batch_size, self.latent_dim)
        :param label: one hot labels with size (batch_size, self.label_dim)
        :return: generated images with size (batch_size, C, H, W), each value is in range [0, 1]
        '''
        with torch.no_grad():
            output = self.forward(z, label)
            output = 0.5 + 0.5 * output

            return output.view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self, label_dim, img_dim, hidden_size=256):
        super(Discriminator, self).__init__()
        self.label_dim = label_dim
        self.img_dim = img_dim
        self.hidden_size = hidden_size
        # Layers
        self.img_fc = nn.Linear(int(np.prod(self.img_dim)), 4 * self.hidden_size)
        self.label_fc = nn.Linear(self.label_dim, 4 * self.hidden_size)
        self.fc2 = nn.Linear(8 * self.hidden_size, 2 * self.hidden_size)
        self.bn2 = nn.BatchNorm1d(2 * self.hidden_size)
        self.fc3 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.bn3 = nn.BatchNorm1d(self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, 1)

        self._initialize(0., 0.02)

    def _initialize(self, mean, std):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()

    def forward(self, img, label):
        img = torch.flatten(img, start_dim=1, end_dim=-1)
        x = nn.functional.leaky_relu(self.img_fc(img), 0.2)
        y = nn.functional.leaky_relu(self.label_fc(label), 0.2)
        x = torch.cat([x, y], dim=-1)
        x = nn.functional.leaky_relu(self.bn2(self.fc2(x)), 0.2)
        x = nn.functional.leaky_relu(self.bn3(self.fc3(x)), 0.2)
        x = torch.sigmoid(self.fc4(x)).squeeze(dim=-1)
        return x

def calc_std(data, label):
    std = torch.zeros((10,))
    for i in range(10):
        std[i] = torch.std(data[label == i], dim=0).mean()
    return std

#########################
####  DO NOT MODIFY  ####
def generate_samples(generator, n_samples_per_class, device):
    generator.eval()
    latent = torch.randn((n_samples_per_class * 10, generator.latent_dim), device=device)
    label = torch.eye(generator.label_dim, dtype=torch.float, device=device).repeat(n_samples_per_class, 1)
    imgs = generator.sample(latent, label).cpu()
    label = torch.argmax(label, dim=-1).cpu()
    samples = dict(imgs=imgs, labels=label)
    return samples
##########################


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    # Load dataset
    if args.dataset == "mnist":
        dataset = MNIST(root="../data",
                        transform=transforms.ToTensor(),  # You can tweak it.
                        train=not args.eval)
        dataloader = DataLoader(dataset, args.batch_size, shuffle=True, drop_last=True)
    else:
        raise NotImplementedError

    # Configure
    logdir = args.logdir if args.logdir is not None else "/tmp/ebm_" + time.strftime('%Y-%m-%d-%H-%M-%S')
    os.makedirs(logdir, exist_ok=True)

    label_dim = 10
    img_dim = (1, 28, 28)
    latent_dim = 100

    device = torch.device("cuda:" + str(args.gpuid) if torch.cuda.is_available() else "cpu")

    generator = Generator(latent_dim, label_dim, img_dim)
    discriminator = Discriminator(label_dim, img_dim)
    generator.to(device)
    discriminator.to(device)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    validator = torch.load("mnist_classifier.pth", map_location=device)

    fake_answer = torch.zeros((args.batch_size,), device=device)

    if not args.eval:
        best_acc = 0

        for epoch in range(args.num_epochs):
            generator.train()
            discriminator.train()

            g_loss_s, d_loss_s = 0, 0
            idx = 0

            pbar = tqdm(total=len(dataloader.dataset))
            for batch_idx, (data, label) in enumerate(dataloader):
                idx = batch_idx
                data = data.to(device)
                label = F.one_hot(label.to(device), 10).float()

                d_loss_s_tmp = 0

                for _ in range(args.d_iter):
                    d_optimizer.zero_grad()

                    d_output = discriminator(data, label)
                    real_answer = 1 - 0.1 * torch.rand((args.batch_size,), device=device)
                    d_loss = nn.BCELoss()(d_output, real_answer) # real
                    d_loss_s_tmp += d_loss.item()

                    z = torch.randn((args.batch_size, generator.latent_dim), device=device)
                    fake_label = F.one_hot(torch.randint(10, size=(args.batch_size,), device=device), 10).float()
                    fake_data = 0.5 + 0.5 * generator(z, fake_label)
                    
                    d_output = discriminator(fake_data, fake_label)
                    d_loss += nn.BCELoss()(d_output, fake_answer) # fake

                    d_loss.backward()
                    d_optimizer.step()
                    d_loss_s_tmp += d_loss.item()
                
                d_loss_s += d_loss_s_tmp / args.d_iter
                
                g_loss_s_tmp = 0
                for _ in range(args.g_iter):
                    g_optimizer.zero_grad()
                    
                    z = torch.randn((args.batch_size, generator.latent_dim), device=device)
                    fake_label = F.one_hot(torch.randint(10, size=(args.batch_size,), device=device), 10).float()
                    fake_data = 0.5 + 0.5 * generator(z, fake_label)
                    
                    d_output = discriminator(fake_data, fake_label)
                    
                    real_answer = 1 - 0.1 * torch.rand((args.batch_size,), device=device)
                    g_loss = nn.BCELoss()(d_output, real_answer) # real
                    g_loss.backward()
                    g_optimizer.step()
                    g_loss_s_tmp += g_loss.item()
                
                g_loss_s += g_loss_s_tmp / args.g_iter

                pbar.update(data.size(0))
                pbar.set_description("Epoch {}. Train, D loss: {:.6f}  G loss: {:.6f}".format(epoch, d_loss_s / (batch_idx + 1), g_loss_s / (batch_idx + 1)))
                pbar.set_postfix(lr=args.lr)
            
            pbar.close()

            logging.info("Epoch {}. Train, D loss: {:.6f}  G loss: {:.6f}".format(epoch, d_loss_s / (idx + 1), g_loss_s / (idx + 1)))

            save_path = os.path.join(args.logdir, "models/model{}.pth".format(epoch))
            state_dict = dict(d=discriminator.state_dict(), g=generator.state_dict(), d_optimizer=d_optimizer.state_dict(), g_optimizer=g_optimizer.state_dict())
            torch.save(state_dict, save_path)
            logging.info('Saving to {}'.format(save_path))

            img_queue = generate_samples(generator, 10, device)['imgs']
            img_grid = torchvision.utils.make_grid(img_queue.cpu(), nrow = 10).numpy()
            plt.imshow(np.transpose(img_grid, (1,2,0)))
            save_path = os.path.join(args.logdir, "images/valid_batch{}.jpg".format(epoch))
            plt.savefig(save_path)

            num_per_class = 1024
            valid_data_queue = generate_samples(generator, num_per_class, device)
            valid_data = valid_data_queue['imgs'].to(device)
            valid_label = valid_data_queue['labels'].to(device)

            std = calc_std(valid_data, valid_label)
            if (std <= std_threshold).sum():
                logging.info("Std <= threshold, valid acc not recorded! {}".format(std))

            acc = 0

            for i in range(num_per_class // 128 * 10):
                data = valid_data[i * 128: (i + 1) * 128, :, :, :]
                data = transforms.Normalize((0.1307,), (0.3081,))(data)
                label = F.one_hot(valid_label[i * 128: (i + 1) * 128], 10).float()
                
                with torch.no_grad():
                    output = validator(data)
                    pred = output.argmax(dim=1, keepdim = True)
                    acc += pred.eq(valid_label[i * 128: (i + 1) * 128].view_as(pred)).sum().item()

            acc /= num_per_class * 10
            logging.info('Valid, acc: {:.6f}'.format(acc))

            if acc > best_acc and (std <= std_threshold).sum() == 0:
                best_acc = acc
                save_path = os.path.join(args.logdir, "models/best.pth")
                torch.save(state_dict, save_path)
                logging.info('Best acc! Saving to {}'.format(save_path))

    else:
        assert args.load_path is not None
        checkpoint = torch.load(args.load_path, map_location=device)
        discriminator.load_state_dict(checkpoint['d'])
        generator.load_state_dict(checkpoint['g'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        # Generate samples for evaluation.
        samples = generate_samples(generator, 1000, device)
        torch.save(samples, "gan_generated_samples.pt")


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
    parser.add_argument("--d_iter", type=int, default=1)
    parser.add_argument("--g_iter", type=int, default=1)
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
