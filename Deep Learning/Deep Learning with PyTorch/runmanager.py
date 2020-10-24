import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from IPython.display import display, clear_output
import pandas as pd
import time
import json

from itertools import product
from collections import namedtuple, OrderedDict

class Epoch:
    def __init(self):
        self.count = 0
        self.loss = 0
        self.correct = 0
        self.start_time = None

class RunManager:
    def __init__(self):
        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        self.network = None
        self.data_loader = None
        self.tb = None

    def begin_run(self, run, network, data_loader, epoch):
        self.run_start_time = time.time()
        epoch.start_time = time.time()
        epoch.count += 1
        epoch.loss = 0
        epoch.correct = 0

        self.run_params = run
        self.run_count += 1

        self.network = network
        self.data_loader = data_loader
        self.tb = SummaryWriter(comment=f'-{run}')

        images, labels = next(iter(self.data_loader))
        grid = torchvision.utils.make_grid(images)
        self.tb.add_image('images', grid)
        self.tb.add_graph(self.network, images)

    def end_run(self, epoch):
        epoch_duration = time.time() - epoch.start_time
        run_duration = time.time() - self.run_start_time

        loss = epoch.loss / len(self.data_loader.dataset)
        accuracy = epoch.correct / len(self.data_loader.dataset)

        self.tb.add_scalar('Loss', loss, epoch.count)
        self.tb.add_scalar('Accuracy', accuracy, epoch.count)

        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, epoch.count)
            self.tb.add_histogram(f'{name}.grad', param.grad, epoch.count)

        results = OrderedDict()
        results['run'] = self.run_count
        results['epoch'] = epoch.count
        results['loss'] = loss
        results['accuracy'] = accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration

        feature_names = list(results.keys())

        for key, value in self.run_params.as_dict().items():
            results[key] = value
        self.run_data.append(results)

        df = pd.DataFrame(
            data=self.run_data,
            columns=feature_names
        )

        clear_output(wait=True)
        display(df)

        self.tb.close()
        epoch.count = 0

    def track_loss(self, loss, batch, epoch):
        epoch.loss += loss.item() * batch[0].shape[0]

    def _get_number_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    def track_num_correct(self, preds, labels, epoch):
        epoch.correct += self._get_number_correct(preds, labels)

    def save(self, filename):
        pd.DataFrame.from_dict(
            self.run_data,
            orient='columns',
        ).to_csv(f'{filename}.csv')

        with open(f'{filename}.json', 'w', encoding='utf=8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)