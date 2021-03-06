from torchvision import transforms
from src.datasets.cifar10 import CIFAR10
from src.datasets.imagenet import ImageNet
from src.datasets.wikipedia import WIKIPEDIA, WIKIPEDIATwoViews
from src.datasets.wikipedia_val import WIKIPEDIAVAL
from src.datasets.newsgroup import NEWSGROUP
from src.datasets.slice import WIKIPEDIASLICE, WIKIPEDIASLICETwoViews

DATASET = {
    'cifar10': CIFAR10,
    'imagenet': ImageNet,
    'wikipedia': WIKIPEDIA,
    'wikipedia_2views': WIKIPEDIATwoViews,
    'wikipedia_val': WIKIPEDIAVAL,
    'newsgroup': NEWSGROUP,
    'slice': WIKIPEDIASLICE,
    'slice_2views': WIKIPEDIASLICETwoViews,
}

def get_datasets(dataset_name, is_wikipedia):
    train_dataset = DATASET[dataset_name](train=True)
    if is_wikipedia:
        val_dataset = DATASET['wikipedia_val'](train=False)
    else:
        val_dataset = DATASET[dataset_name](train=False)
    return train_dataset, val_dataset


def load_default_transforms(dataset):
    # resize imagenet to 256
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]),
    ])
    return train_transforms, test_transforms
