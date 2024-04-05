from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, ToTensor, ToPILImage, CenterCrop, Resize, Normalize


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir):
        super(TrainDatasetFromFolder, self).__init__()
        self.hr_dataset_dir = dataset_dir + "/hr_images"
        self.hr_img_files = load_pngs(self.hr_dataset_dir)
        self.hr_transform = to_tensor()

        self.lr_dataset_dir = dataset_dir + "/lr_images"
        self.lr_img_files = load_pngs(self.lr_dataset_dir)
        self.lr_transform = to_tensor()

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.hr_img_files[index]))
        lr_image = self.lr_transform(Image.open(self.lr_img_files[index]))
        return lr_image, hr_image

    def __len__(self):
        return len(self.hr_img_files)


class ValidationDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir):
        super(ValidationDatasetFromFolder, self).__init__()
        self.hr_dataset_dir = dataset_dir + "/hr_images"
        self.hr_img_files = load_pngs(self.hr_dataset_dir)
        self.hr_transform = to_tensor()

        self.lr_dataset_dir = dataset_dir + "/lr_images"
        self.lr_img_files = load_pngs(self.lr_dataset_dir)
        self.lr_transform = to_tensor()

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.hr_img_files[index]))
        lr_image = self.lr_transform(Image.open(self.lr_img_files[index]))
        return lr_image, hr_image

    def __len__(self):
        return len(self.hr_img_files)


def to_tensor():
    return Compose([ToTensor()])


def load_pngs(dataset_dir):
    return [join(dataset_dir, file) for file in listdir(dataset_dir) if file.endswith('.png')]


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])

