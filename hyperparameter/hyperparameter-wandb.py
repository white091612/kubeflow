import argparse
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import timm
from torch.optim.lr_scheduler import CosineAnnealingLR
from util import seed_everything, MetricMonitor, build_dataset, build_optimizer
import wandb
from lion_pytorch import Lion


class wandbModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super(wandbModel, self).__init__()
        self.model = timm.create_model(model_name=model_name, pretrained=pretrained,
                                       num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


def train_epoch(train_loader, epoch, model, optimizer, criterion, device):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    for i, (images, targets) in enumerate(stream, start=1):
        images, targets = images.float().to(device), targets.to(device)
        output = model(images)
        loss = criterion(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        predicted = torch.argmax(output, dim=1)
        accuracy = round((targets == predicted).sum().item() / targets.shape[0] * 100, 2)
        metric_monitor.update('Loss', loss.item())
        metric_monitor.update('accuracy', accuracy)
        stream.set_description(
            f"Epoch: {epoch}. Train. {metric_monitor}"
        )
        wandb.log({"Train Epoch": epoch, "Train loss": loss.item(), "Train accuracy": accuracy})


def val_epoch(val_loader, epoch, model, criterion, device):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    val_loss = 0
    with torch.no_grad():
        for i, (images, targets) in enumerate(stream, start=1):
            images, targets = images.float().to(device), targets.to(device)
            output = model(images)
            loss = criterion(output, targets)
            val_loss += loss
            predicted = torch.argmax(output, dim=1)
            accuracy = round((targets == predicted).sum().item() / targets.shape[0] * 100, 2)
            metric_monitor.update('Loss', loss.item())
            metric_monitor.update('accuracy', accuracy)
            stream.set_description(
                f"Epoch: {epoch}. Validation. {metric_monitor}"
            )
            wandb.log({"Validation Epoch": epoch, "Validation loss": loss.item(), "Validation accuracy": accuracy})
        wandb.log({"VAL EPOCH LOSS": val_loss / len(val_loader.dataset)})
    return accuracy


def main(hyperparameters=None):
    wandb.init(project='skin-classification', config=hyperparameters)
    config = wandb.config
    epochs = 3
    # read mean std values
    with open(f'{opt.data_path}/mean-std.txt', 'r') as f:
        cc = f.readlines()
        mean_std = list(map(lambda x: x.strip('\n'), cc))
    # model = wandbModel(num_classes=2, pretrained=True, model_name=config.model)
    model = wandbModel(num_classes=7, pretrained=True, model_name=config.model)
    model.to(device)
    depth = 0
    for name, param in model.named_parameters():
        depth += 1
    cnt = 0
    for name, param in model.named_parameters():
        cnt += 1
        if cnt < int(depth * config.freeze):
            param.requires_grad = False
    train_loader, val_loader, _ = build_dataset(opt.data_path, config.img_size, config.batch_size, mean_std)
    optimizer = build_optimizer(model, config.optimizer, config.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=10,
                                  eta_min=1e-6,
                                  last_epoch=-1)
    for epoch in range(1, epochs + 1):
        train_epoch(train_loader, epoch, model, optimizer, criterion, device)
        val_epoch(val_loader, epoch, model, criterion, device)
        scheduler.step()


def configure():

    sweep_config = \
        {'method': 'random',
         'metric': {'goal': 'minimize', 'name': 'VAL EPOCH LOSS'},
         # 'parameters': {'model': {'values': ["swinv2_cr_small_224",
         #                                     "coatnet_rmlp_2_rw_224",
         #                                     "maxvit_tiny_rw_224"]
         #                          },
         'parameters': {'model': {'values': ["tf_efficientnetv2_l","swinv2_cr_small_224"]},
                        'batch_size': {'value': 32},
                        'epochs': {'value': 5},
                        'img_size': {'value': 224},
                        'lr': {'distribution': 'uniform',
                               'max': 0.005,
                               'min': 0.0001},
                        'freeze': {'values': [0.9, 0.95]},
                        'optimizer': {'values': ['adam', 'sgd', 'lion']}
                        }
         }
    return sweep_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, help='dataset root path')
    parser.add_argument('--device', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    if opt.device == 'cpu':
        device = 'cpu'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.device
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DEVICE is {device}")
    seed_everything()
    wandb.login(key='69101e4d701cdeab6822d00f3a898ff567bd706c')
    hyperparameters = configure()
    sweep_id = wandb.sweep(hyperparameters, project='skin-classification')
    wandb.agent(sweep_id, main, count=60)  # count: 실험 횟수
