import torchvision.transforms as transforms
import torchvision.datasets as dset


def cifar_transform(is_training=True):
  if is_training:
    transform_list = [transforms.RandomHorizontalFlip(),
                      transforms.Pad(padding=4, padding_mode='reflect'),
                      transforms.RandomCrop(32, padding=0),
                      transforms.ToTensor(),
                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ]
  else:
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ]

  transform_list = transforms.Compose(transform_list)
  return transform_list


def imgnet_transform(is_training=True):
  if is_training:
    transform_list = transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ColorJitter(brightness=0.5,
                                                                contrast=0.5,
                                                                saturation=0.3),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
  else:
    transform_list = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
  return transform_list

def coco_transform(is_training=True):
    if is_training:
        transform_list = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform_list = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform_list


def voc_transform(is_training=True):
    if is_training:
        transform_list = [
            transforms.Resize((448, 448)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    else:
        transform_list = [
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

    transform_list = transforms.Compose(transform_list)
    return transform_list

def create_voc_datasets(data_dir, is_training):
    transform = voc_transform(is_training)
    voc_dataset = dset.VOCDetection(
        root=data_dir, year="2012", image_set="train" if is_training else "val", transform=transform, download=False, 
    )
    return voc_dataset
