import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class FER2013(Dataset):
    def __init__(self, csv_file, split="Train", transform=None):

        self.split = str(split.upper())
        if self.split not in {"TRAIN", "PUBLIC_TEST", "PRIVATE_TEST"}:
            print("Param split not in {TRAIN, PUBLIC_TEST, PRIVATE_TEST}")
            assert self.split in {"TRAIN", "PUBLIC_TEST", "PRIVATE_TEST"}

        dataset = pd.read_csv(csv_file)
        # print(dataset.head(2))

        self.transform = transform
        if self.split == "TRAIN":
            # self.data = dataset[dataset["Usage"] == "Training"]
            self.data = dataset[dataset.iloc[:, 1] == "Training"]
            # assert len(self.data) == 28709
        elif self.split == "PUBLIC_TEST":
            # self.data = dataset[dataset["Usage"] == "PublicTest"]
            self.data = dataset[dataset.iloc[:, 1] == 'PublicTest']
            # assert len(self.data) == 3589
        else:
            # self.data = dataset[dataset["Usage"] == "PrivateTest"]
            self.data = dataset[dataset.iloc[:, 1] == "PrivateTest"]
            # assert len(self.data) == 3589

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # image = list(map(int, self.data["pixels"].iloc[idx].split(" ")))
        image = list(map(int, self.data.iloc[idx, 2].split(" ")))

        image = np.array(image)
        image = image.reshape(48, 48).astype(np.uint8)

        # image = image[:, :, np.newaxis]
        # image = np.concatenate((image, image, image), axis= 2)
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        target = self.data.iloc[idx, 0]
        return image, target
