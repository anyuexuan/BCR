import numpy as np
import os
from torch.utils.data import Dataset
import torch
from PIL import Image
from scipy.sparse import load_npz
from torchvision import transforms

current_path = os.path.dirname(__file__)

path_to_images_dict = {
    'CUB': os.path.join(current_path, '../../data/MLL/CUB/CUB_200_2011/images'),
    'COCO': os.path.join(current_path, '../../data/MLL/COCO2014'),
    'NUSWIDE': os.path.join(current_path, '../../data/MLL/NUSWIDE/Flickr'),
    'VG': os.path.join(current_path, '../../data/MLL/VG/VG_100K'),
}


class MLLDataset(Dataset):
    def __init__(self, dataset_name, phase='train', transform=True, image_size=84):
        self.dataset_name = dataset_name
        self.images = np.load(os.path.join(current_path, 'idx', f'{dataset_name}_{phase}_images.npy'))
        self.labels = load_npz(
            os.path.join(current_path, 'idx', f'{dataset_name}_{phase}_labels.npz')).toarray()
        self.image_transform = get_transform(transform=transform, image_size=image_size)
        self.max_idx = self.images.shape[0]
        self.image_size = image_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if index < self.labels.shape[0]:
            image = Image.open(os.path.join(path_to_images_dict[self.dataset_name], self.images[index])).convert('RGB')
            image = self.image_transform(image)
            labels = torch.Tensor(self.labels[index])
            sample = {}
            sample['image'] = image
            sample['labels'] = labels
            sample['idx'] = index
        else:
            sample = {}
            sample['image'] = torch.zeros([3, self.image_size, self.image_size])
            sample['labels'] = torch.zeros([self.labels.shape[1]])
            sample['labels'][index - self.max_idx] = 1
            sample['idx'] = index
        return sample


class EpisodeSampler:
    def __init__(self, dataset_name, n_way, n_shot, max_idx,
                 n_query=16, phase='val', iter=100):
        self.dataset_name = dataset_name
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.labels = load_npz(
            os.path.join(current_path, 'idx', f'{dataset_name}_{phase}_labels.npz')).toarray()
        assert self.labels.shape[1] >= self.n_way  # number of ways should not more than number of labels
        self.images = np.load(os.path.join(current_path, 'idx', f'{dataset_name}_{phase}_images.npy'))
        self.label_to_idx_dict = \
            np.load(os.path.join(current_path, 'idx', f'{dataset_name}_' + phase + '_label_to_idx_dict.npy'),
                    allow_pickle=True)[0]
        self.iter = iter
        self.max_idx = max_idx

    def __len__(self):
        return self.iter

    def __iter__(self):
        for _ in range(self.iter):
            # ------------ sample classes ------------
            sampled_class = np.random.choice(np.arange(self.labels.shape[1]), size=self.n_way, replace=False)
            # ------------ sample samples ------------
            support_idx = []
            query_idx = []
            for c in sampled_class:
                idx = np.random.choice(self.label_to_idx_dict[c],
                                       size=np.minimum(self.n_shot + self.n_query, len(self.label_to_idx_dict[c])),
                                       replace=False)
                support_idx.extend(idx[:self.n_shot])
                query_idx.extend(idx[self.n_shot:])

            last_set = list(set(query_idx) - set(support_idx))
            query_idx = np.random.choice(list(last_set), size=self.n_query, replace=False)
            class_idx = self.max_idx + sampled_class
            if self.n_query == 0:
                all_idx = np.concatenate([support_idx, class_idx])
            else:
                all_idx = np.concatenate([support_idx, query_idx, class_idx])
            assert len(all_idx) == self.n_way * self.n_shot + self.n_query + self.n_way
            yield all_idx


def generate_MetaDataset(dataset_name, n_way, n_shot, image_size=224,
                         n_query=16, phase='val', transform=False):
    images = np.load(os.path.join(current_path, 'idx', f'{dataset_name}_{phase}_images.npy'))
    labels = load_npz(os.path.join(current_path, 'idx', f'{dataset_name}_{phase}_labels.npz')).toarray()
    assert labels.shape[1] >= n_way  # number of ways should not more than number of labels
    # ------------ sample classes ------------
    sampled_class = np.random.choice(np.arange(labels.shape[1]), size=n_way, replace=False)
    # ------------ sample samples ------------
    support_idx = []
    query_idx = []
    label_to_idx_dict = \
        np.load(os.path.join(current_path, 'idx', f'{dataset_name}_' + phase + '_label_to_idx_dict.npy'),
                allow_pickle=True)[0]
    for c in sampled_class:
        idx = np.random.choice(label_to_idx_dict[c], size=np.minimum(n_shot + n_query, len(label_to_idx_dict[c])),
                               replace=False)
        support_idx.extend(idx[:n_shot])
        query_idx.extend(idx[n_shot:])
    support_idx = np.array(list(set(support_idx)))
    query_idx = np.array(list(set(query_idx) - set(support_idx)))
    # ------------ load images ------------
    image_transform = get_transform(transform=transform, image_size=image_size)
    x_support = torch.zeros([len(support_idx), 3, image_size, image_size])
    for i, s in enumerate(support_idx):
        image = Image.open(os.path.join(path_to_images_dict[dataset_name], images[s])).convert('RGB')
        x_support[i] = image_transform(image)
    x_query = torch.zeros([len(query_idx), 3, image_size, image_size])
    for i, q in enumerate(query_idx):
        image = Image.open(os.path.join(path_to_images_dict[dataset_name], images[q])).convert('RGB')
        x_query[i] = image_transform(image)
    # ------------ load labels ------------
    y_support = torch.from_numpy(labels[support_idx][:, sampled_class])
    y_query = torch.from_numpy(labels[query_idx][:, sampled_class])
    test_loader = (x_support, y_support, x_query, y_query)
    return test_loader


def get_transform(transform, image_size):
    if transform:
        return transforms.Compose([
            transforms.RandomResizedCrop((image_size, image_size)),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
    else:
        return transforms.Compose([
            transforms.Resize((int(image_size * 1.15), int(image_size * 1.15))),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
