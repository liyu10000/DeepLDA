import os
import random
import numpy as np
np.set_printoptions(precision=4, suppress=True)
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from functools import partial
from lda import LDA, lda_loss


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, n_classes, lda_args):
        super(ResNet, self).__init__()
        self.lda_args = lda_args
        if self.lda_args:  # LDA
            self.in_planes = 32
            self.out_planes = 16
        else:  # Usual CNN with CE loss
            self.in_planes = 32
            self.out_planes = 16 # 64

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, self.out_planes*1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, self.out_planes*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, self.out_planes*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, self.out_planes*8, num_blocks[3], stride=2)
        if self.lda_args:
            self.lda = LDA(n_classes, lda_args['lamb'])
        else:
            self.linear = nn.Linear(self.out_planes*8*block.expansion, n_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, X, y):
        out = F.relu(self.bn1(self.conv1(X)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        fea = out.view(out.size(0), -1)  # NxC
        if self.lda_args:
            hasComplexEVal, out = self.lda(fea, y)  # evals
            return hasComplexEVal, fea, out
        else:
            out = self.linear(fea)
            return out


def ResNet18(n_classes, lda_args):
    return ResNet(BasicBlock, [2, 2, 2, 2], n_classes, lda_args)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


class CIFAR10:
    def __init__(self, img_names, class_map, transform):
        self.img_names = img_names
        self.classes = [class_map[os.path.basename(os.path.dirname(n))] for n in img_names]
        self.transform = transform
    def __len__(self):
        return len(self.img_names)
    def __getitem__(self, idx):
        img = Image.open(self.img_names[idx])
        img = self.transform(img)
        clazz = self.classes[idx]
        return img, clazz


class Solver:
    def __init__(self, dataloaders, model_path, n_classes, lda_args={}, gpu=-1):
        self.dataloaders = dataloaders
        if gpu >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu:0')
        self.net = ResNet18(n_classes, lda_args)
        self.net = self.net.to(self.device)
        self.use_lda = True if lda_args else False
        if self.use_lda:
            self.criterion = partial(lda_loss, n_classes=n_classes, 
                                    n_eig=lda_args['n_eig'], margin=lda_args['margin'])
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        self.model_path = model_path
        self.n_classes = n_classes

    def iterate(self, epoch, phase):
        self.net.train(phase == 'train')
        dataloader = self.dataloaders[phase]
        total_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad()
            
            if self.use_lda:
                hasComplexEVal, feas, outputs = self.net(inputs, targets)
                if not hasComplexEVal:
                    loss = self.criterion(outputs)
                    outputs = self.net.lda.predict_proba(feas)
                else:
                    print('Complex Eigen values found, skip backpropagation of {}th batch'.format(batch_idx))
                    continue
            else:
                outputs = self.net(inputs, targets)
                loss = self.criterion(outputs, targets)            
            # print('\noutputs shape:', outputs.shape)
            # print('loss:', loss)
            if phase == 'train':
                loss.backward()
                self.optimizer.step()
            total_loss += loss.item()

            outputs = torch.argmax(outputs.detach(), dim=1)
            # _, outputs = outputs.max(1)
            total += targets.size(0)
            correct += outputs.eq(targets).sum().item()
        total_loss /= (batch_idx + 1)
        total_acc = correct/total
        print('\nepoch %d: %s loss: %.3f | acc: %.2f%% (%d/%d)'
                     % (epoch, phase, total_loss, 100.*total_acc, correct, total))
        return total_loss, total_acc

    def train(self, epochs):
        best_loss = float('inf')
        for epoch in range(epochs):
            self.iterate(epoch, 'train')
            with torch.no_grad():
                val_loss, val_acc = self.iterate(epoch, 'val')
            if val_loss < best_loss:
                best_loss = val_loss
                checkpoint = {'epoch':epoch, 'val_loss':val_loss, 'state_dict':self.net.state_dict()}
                print('best val loss found')
            print()
        torch.save(checkpoint, self.model_path)

    def test_iterate(self, epoch, phase):
        self.net.eval()
        dataloader = self.dataloaders[phase]
        y_pred = []
        y_true = []
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                if self.use_lda:
                    _, feas, outputs = self.net(inputs, targets)
                    outputs = self.net.lda.predict_proba(feas)
                else:
                    outputs = self.net(inputs, targets)
                outputs = torch.argmax(outputs, dim=1)
                y_pred.append(outputs.detach().cpu().numpy())
                y_true.append(targets.detach().cpu().numpy())
        return np.array(y_pred).flatten(), np.array(y_true).flatten()
        
    def test(self):
        checkpoint = torch.load(self.model_path)
        epoch = checkpoint['epoch']
        val_loss = checkpoint['val_loss']
        self.net.load_state_dict(checkpoint['state_dict'])
        print('load model at epoch {}, with val loss: {:.3f}'.format(epoch, val_loss))
        y_pred, y_true = self.test_iterate(epoch, 'test')
        print(y_pred.shape, y_true.shape)

        print('total', accuracy_score(y_true, y_pred))
        for i in range(self.n_classes):
            idx = y_true == i
            print('class', i, accuracy_score(y_true[idx], y_pred[idx]))


def parse_dir(img_dir, classes, randnum=-1):
    img_names = []
    ids = []
    for clazz in classes:
        sub_dir = os.path.join(img_dir, clazz)
        sub_files = [os.path.join(sub_dir, f) for f in os.listdir(sub_dir)]
        if len(sub_files) > randnum > 0:
            sub_files = random.sample(sub_files, randnum)
        img_names += sub_files
    for img_name in img_names:
        clazz = os.path.basename(os.path.dirname(img_name))
        id = clazz + '+' + os.path.basename(img_name)
        ids.append(id)
    return ids


if __name__ == '__main__':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    seed = 42
    n_classes = 10
    train_val_split = 0.2
    batch_size = 5000
    num_workers = 4
    gpu = -1

    train_dir = '../data/cifar10/imgs/train'
    test_dir = '../data/cifar10/imgs/test'
    model_path = '../data/cifar10/exp1015/deeplda_best.pth'

    loss = 'LDA' # CE or LDA
    lamb = 0.0001
    n_eig = 4
    margin = None
    lda_args = {'lamb':lamb, 'n_eig':n_eig, 'margin':margin} if loss == 'LDA' else {}

    class_map = {'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4, 
                 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}
    
    ids = parse_dir(train_dir, os.listdir(train_dir))
    train_img_names = [os.path.join(train_dir, *f.split('+')) for f in ids]
    trainset = CIFAR10(train_img_names, class_map, transform_train)
    N = len(trainset)
    Ntrain, Nval = N - int(N * train_val_split), int(N * train_val_split)
    trainset, valset = torch.utils.data.random_split(trainset, [Ntrain, Nval])
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_ids = parse_dir(test_dir, os.listdir(test_dir))
    test_img_names = [os.path.join(test_dir, *f.split('+')) for f in test_ids]
    testset = CIFAR10(test_img_names, class_map, transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    dataloaders = {'train':trainloader, 'val':valloader, 'test':testloader}
    solver = Solver(dataloaders, model_path, n_classes, lda_args, gpu)
    solver.train(20)
    solver.test()
