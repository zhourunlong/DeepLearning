import argparse, time
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from ebm import MlpBackbone, Trainer


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    # Load dataset
    if args.dataset == "mnist":
        dataset = MNIST(root="../data", transform=transforms.ToTensor(), train=not args.play)
        dataloader = DataLoader(dataset, args.batch_size, shuffle=True, drop_last=True)
    else:
        raise NotImplementedError
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Model
    model = MlpBackbone(input_shape=(1, 28, 28), hidden_size=512)
    model.to(device)
    # EBM trainer
    trainer = Trainer(model, device, args.buffer_size, args.langevin_k, args.langevin_noise_std,
                      args.langevin_lr, args.replay_p, args.lr, args.l2_coef, args.proj_norm)

    if not args.play:
        # Training.
        for epoch in range(args.num_epochs):
            for data, label in dataloader:
                # TODO: add logging, saving, or anything else you believe useful
                # trainer.train_step(data.to(device))
                pass
    else:
        # Play around with a trained model.
        assert args.load_dir is not None, "You must specify load_dir."
        trainer.load(args.load_dir)
        all_mse = []
        initial_mse = []
        for data, label in dataloader:
            broken_data = torch.clone(data)
            # Corrupt the rows 0, 2, 4, ....
            mask = torch.zeros_like(broken_data, device=device).bool()
            for i in range(0, mask.shape[-2], 2):
                mask[:, :, i, :] = True
            broken_data = broken_data.to(device)
            broken_data[mask] += 0.3 * torch.randn(*broken_data.shape, device=device)[mask]
            broken_data = torch.clip(broken_data, 0., 1.)
            # TODO: just an example. You may add more visualization or anything else.
            recovered_img = trainer.inpainting(broken_data.to(device), mask)
            mse = np.mean((data.numpy() - recovered_img) ** 2, axis=(1, 2, 3))
            all_mse.extend(mse.tolist())
            mse = np.mean((data.numpy() - broken_data.cpu().numpy()) ** 2, axis=(1, 2, 3))
            initial_mse.extend(mse.tolist())
        print("mean mse", np.mean(all_mse), "std", np.std(all_mse), "max", np.max(all_mse), "min", np.min(all_mse))
        print("mean mse", np.mean(initial_mse), "std", np.std(initial_mse), "max", np.max(initial_mse), "min", np.min(initial_mse))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", choices=["mnist"], default="mnist")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--play", action="store_true", default=False)
    parser.add_argument("--load_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--langevin_k", type=int, default=1)
    parser.add_argument("--langevin_noise_std", type=float, default=1)
    parser.add_argument("--langevin_lr", type=float, default=1)
    parser.add_argument("--replay_p", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1)
    parser.add_argument("--l2_coef", type=float, default=0)
    parser.add_argument("--proj_norm", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=10)
    args = parser.parse_args()
    main(args)
