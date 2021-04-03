import os

import itertools
import time
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import optim, nn
from torchvision import transforms
from torchvision.utils import make_grid

from .conf import CycleGAN_Settings as Settings
from .models.Discriminator import Discriminator
from .models.Generator import Generator
from utils.DataLoaders import get_dataloader


class CycleGAN(object):

    def __init__(self):
        # ------
        # global
        # ------
        np.random.seed(Settings.SEED)
        torch.manual_seed(Settings.SEED)
        random.seed(Settings.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(Settings.SEED)
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # ------
        # model
        # ------
        self.generatorA2B = Generator()
        self.generatorB2A = Generator()
        self.discriminatorA = Discriminator()
        self.discriminatorB = Discriminator()
        print("models init done .....")

        # ------
        # data
        # ------
        train_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        data_prepare = get_dataloader(dataset_name=Settings.DATASET,
                                      batch_size=Settings.BATCH_SIZE,
                                      data_root=Settings.DATASET_ROOT,
                                      train_num_workers=Settings.TRAIN_NUM_WORKERS,
                                      transforms=train_transforms,
                                      val_num_workers=Settings.TEST_NUM_WORKERS)
        self.train_dataloader = data_prepare.train_dataloader
        self.test_dataloader = data_prepare.test_dataloader
        print("data init done.....")

        # ------
        # optimizer and criterion
        # ------
        self.optimG = optim.Adam(itertools.chain(self.generatorA2B.parameters(), self.generatorB2A.parameters()),
                                 lr=Settings.G_LR, betas=Settings.G_BETAS)
        self.optimD = optim.Adam(itertools.chain(self.discriminatorA.parameters(), self.discriminatorB.parameters()),
                                 lr=Settings.D_LR, betas=Settings.D_BETAS)

        self.criterion_cycle = nn.L1Loss()
        self.criterion_idt = nn.L1Loss()
        self.criterion_BCE = nn.BCELoss()
        print("optimizer and criterion init done.....")

        # ------
        # recorder
        # ------
        self.recorder = {"errD_A": list(),
                         "errD_B": list(),
                         "errG_A": list(),
                         "errG_B": list(),
                         "errD_A_fake": list(),
                         "errD_A_real": list(),
                         "errD_B_fake": list(),
                         "errD_B_real": list(),
                         "errG_A2B_bce": list(),
                         "errG_B2A_bce": list(),
                         "errG_A2B2A_cycle": list(),
                         "errG_B2A2B_cycle": list(),
                         "accD_A": list(),
                         "accD_B": list()}
        if Settings.USING_IDENTITY_LOSS:
            self.recorder["errG_A_idt"] = list()
            self.recorder["errG_B_idt"] = list()

        output_file = time.strftime("{}_{}_%Y_%m_%d_%H_%M_%S".format("CycleGAN", Settings.DATASET), time.localtime())
        self.output_root = os.path.join(Settings.OUTPUT_ROOT, output_file)
        os.makedirs(os.path.join(self.output_root, Settings.OUTPUT_MODEL_KEY))
        os.makedirs(os.path.join(self.output_root, Settings.OUTPUT_LOG_KEY))
        os.makedirs(os.path.join(self.output_root, Settings.OUTPUT_IMAGE_KEY))
        print("recorder init done.....")

    def __call__(self):
        pass

    def train_module(self, batch):
        pass

    def backward_discriminator(self):
        pass

    def eval_module(self):
        pass

    def print_module(self):
        pass

    def model_save_module(self):
        pass

    def log_save_module(self):
        pass

    def learning_rate_decay_module(self):
        pass


if __name__ == "__main__":
    cycle_gan = CycleGAN()
    cycle_gan()

