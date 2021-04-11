import argparse, time, logging, os, sys
import torch, torchvision
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from ebm import MlpBackbone, Trainer
from collections import deque
from PIL import Image
import matplotlib.pyplot as plt

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
    device = torch.device("cuda:"+str(args.gpuid) if torch.cuda.is_available() else "cpu")
    # Model
    model = MlpBackbone(input_shape=(1, 28, 28), hidden_size=512)
    model.to(device)
    # EBM trainer
    trainer = Trainer(model, device, args.seed, args.buffer_size, args.langevin_k, args.langevin_noise_std, args.langevin_lr, args.replay_p, args.lr, args.l2_coef, args.proj_norm)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.logdir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    if not args.play:
        # Training.
        for epoch in range(args.num_epochs):
            cnt = 0
            running_loss = 0
            for pos_img, label in dataloader:
                loss = trainer.train_step(pos_img.to(device))

                cnt += 1
                running_loss += loss
                
                if cnt % args.log_interval == 0:
                    logging.info("epoch %d, average loss %f" % (epoch + 1, running_loss / cnt))
            
            save_path = os.path.join(args.logdir, "ebm_checkpoint_{}.pth".format(epoch))
            trainer.save(save_path)
            logging.info("saved checkpoint to {}".format(save_path))

    else:
        # Play around with a trained model.
        assert args.load_dir is not None, "You must specify load_dir."
        trainer.load(args.load_dir)
        all_mse = []
        initial_mse = []
        cnt = 0
        for data, label in dataloader:
            broken_data = torch.clone(data)
            # Corrupt the rows 0, 2, 4, ....
            mask = torch.zeros_like(broken_data, device=device).bool()
            for i in range(0, mask.shape[-2], 2):
                mask[:, :, i, :] = True
            broken_data = broken_data.to(device)
            broken_data[mask] += 0.3 * torch.randn(*broken_data.shape, device=device)[mask]
            broken_data = torch.clip(broken_data, 0., 1.)

            recovered_img = trainer.inpainting(broken_data.to(device), mask)

            img_queue = []
            for i in range(data.shape[0]):
                img_queue.append(broken_data[i])
                img_queue.append(recovered_img[i])
            img = torchvision.utils.make_grid(torch.stack(img_queue).cpu(), nrow = 16).numpy()
            plt.imshow(np.transpose(img, (1,2,0)))
            save_path = os.path.join(args.logdir, "denoise_batch{}.jpg".format(cnt))
            plt.savefig(save_path)
            logging.info("saved {}".format(save_path))

            mse = np.mean((data.numpy() - recovered_img.cpu().numpy()) ** 2, axis=(1, 2, 3))
            all_mse.extend(mse.tolist())
            mse = np.mean((data.numpy() - broken_data.cpu().numpy()) ** 2, axis=(1, 2, 3))
            initial_mse.extend(mse.tolist())

            cnt += 1
        logging.info("all:     mean mse %f std %f max %f min %f" % (np.mean(all_mse), np.std(all_mse), np.max(all_mse), np.min(all_mse)))
        logging.info("initial: mean mse %f std %f max %f min %f" % (np.mean(initial_mse), np.std(initial_mse), np.max(initial_mse), np.min(initial_mse)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", choices=["mnist"], default="mnist")
    parser.add_argument("--seed", type=int, default=2018011309)
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--play", action="store_true", default=False)
    parser.add_argument("--load_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--langevin_k", type=int, default=60)
    parser.add_argument("--langevin_noise_std", type=float, default=0.005)
    parser.add_argument("--langevin_lr", type=float, default=10)
    parser.add_argument("--replay_p", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--l2_coef", type=float, default=1)
    parser.add_argument("--proj_norm", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--gpuid", default=0, type=int)
    args = parser.parse_args()

    if args.logdir is None:
        args.logdir = "Models-{}".format(time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(args.logdir, exist_ok=True)
    print("Experiment dir : {}".format(args.logdir))

    main(args)
