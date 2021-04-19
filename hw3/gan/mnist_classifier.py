from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

class MnistClassifier(nn.Module):
    def __init__(self):
        super(MnistClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, padding=1) 
        self.bn1 = nn.BatchNorm2d(num_features=128)

        self.tns1 = nn.Conv2d(in_channels=128, out_channels=8, kernel_size=1, padding=1)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1) 
        self.bn2 = nn.BatchNorm2d(num_features=16)  
        self.pool1 = nn.MaxPool2d(2, 2)   
        

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1) 
        self.bn3 = nn.BatchNorm2d(num_features=16) 
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1) 
        self.bn4 = nn.BatchNorm2d(num_features=32)
        self.pool2 = nn.MaxPool2d(2, 2) 

        self.tns2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1) 
        self.bn5 = nn.BatchNorm2d(num_features=16) 
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1) 
        self.bn6 = nn.BatchNorm2d(num_features=32)

        self.conv7 = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=1, padding=1)    
        
        self.gpool = nn.AvgPool2d(kernel_size=7)

        self.drop = nn.Dropout2d(0.1)

    def forward(self, x):
        x = self.tns1(self.drop(self.bn1(F.relu(self.conv1(x)))))
        x = self.drop(self.bn2(F.relu(self.conv2(x))))

        x = self.pool1(x)

        x = self.drop(self.bn3(F.relu(self.conv3(x))))        
        x = self.drop(self.bn4(F.relu(self.conv4(x))))

        x = self.tns2(self.pool2(x))

        x = self.drop(self.bn5(F.relu(self.conv5(x))))
        x = self.drop(self.bn6(F.relu(self.conv6(x))))

        x = self.conv7(x)
        x = self.gpool(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pbar.set_description(desc= f'epoch: {epoch} loss={loss.item()} batch_id={batch_idx}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return correct

if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    torch.manual_seed(1)
    batch_size = 128

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    model = MnistClassifier().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    best_acc = 0

    for epoch in range(1, 100):
        train(model, device, train_loader, optimizer, epoch)
        acc = test(model, device, test_loader)

        if acc > best_acc:
            best_acc = acc
            print("saving...")
            torch.save(model, 'best_mnist_model.pth')
