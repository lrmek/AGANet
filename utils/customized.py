"""
Customized dataset
"""

import os
import random
# from random import choice

import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO

class BaseDataset(Dataset):
    """
    Base Dataset

    Args:
        base_dir:
            dataset directory
    """
    def __init__(self, base_dir):
        self._base_dir = base_dir
        self.aux_attrib = {}
        self.aux_attrib_args = {}
        self.ids = []  # must be overloaded in subclass

    def add_attrib(self, key, func, func_args):
        """
        Add attribute to the data sample dict

        Args:
            key:
                key in the data sample dict for the new attribute
                e.g. sample['click_map'], sample['depth_map']
            func:
                function to process a data sample and create an attribute (e.g. user clicks)
            func_args:
                extra arguments to pass, expected a dict
        """
        if key in self.aux_attrib:
            raise KeyError("Attribute '{0}' already exists, please use 'set_attrib'.".format(key))
        else:
            self.set_attrib(key, func, func_args)

    def set_attrib(self, key, func, func_args):
        """
        Set attribute in the data sample dict

        Args:
            key:
                key in the data sample dict for the new attribute
                e.g. sample['click_map'], sample['depth_map']
            func:
                function to process a data sample and create an attribute (e.g. user clicks)
            func_args:
                extra arguments to pass, expected a dict
        """
        self.aux_attrib[key] = func
        self.aux_attrib_args[key] = func_args

    def del_attrib(self, key):
        """
        Remove attribute in the data sample dict

        Args:
            key:
                key in the data sample dict
        """
        self.aux_attrib.pop(key)
        self.aux_attrib_args.pop(key)

    def subsets(self, sub_ids, sub_args_lst=None):
        """
        Create subsets by ids

        Args:
            sub_ids:
                a sequence of sequences, each sequence contains data ids for one subset
            sub_args_lst:
                a list of args for some subset-specific auxiliary attribute function
        """

        indices = [[self.ids.index(id_) for id_ in ids] for ids in sub_ids]
        if sub_args_lst is not None:
            subsets = [Subset(dataset=self, indices=index, sub_attrib_args=args)
                       for index, args in zip(indices, sub_args_lst)]
        else:
            subsets = [Subset(dataset=self, indices=index) for index in indices]
        return subsets

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

class PairedDataset(Dataset):
    """
    Make pairs of data from dataset

    When 'same=True',
        a pair contains data from same datasets,
        and the choice of datasets for each pair is random.
        e.g. [[ds1_3, ds1_2], [ds3_1, ds3_2], [ds2_1, ds2_2], ...]
    When 'same=False',
            a pair contains data from different datasets,
            if 'n_elements' <= # of datasets, then we randomly choose a subset of datasets,
                then randomly choose a sample from each dataset in the subset
                e.g. [[ds1_3, ds2_1, ds3_1], [ds4_1, ds2_3, ds3_2], ...]
            if 'n_element' is a list of int, say [C_1, C_2, C_3, ..., C_k], we first
                randomly choose k(k < # of datasets) datasets, then draw C_1, C_2, ..., C_k samples
                from each dataset respectively.
                Note the total number of elements will be (C_1 + C_2 + ... + C_k).

    Args:
        datasets:
            source datasets, expect a list of Dataset
        n_elements:
            number of elements in a pair
        max_iters:
            number of pairs to be sampled
        same:
            whether data samples in a pair are from the same dataset or not,
            see a detailed explanation above.
        pair_based_transforms:
            some transformation performed on a pair basis, expect a list of functions,
            each function takes a pair sample and return a transformed one.
    """
    def __init__(self, datasets, n_elements, max_iters, same=True,
                 pair_based_transforms=None):
        super(PairedDataset, self).__init__()
        self.datasets = datasets
        self.n_datasets = len(self.datasets)
        self.n_data = [len(dataset) for dataset in self.datasets]
        self.n_elements = n_elements
        self.max_iters = max_iters
        self.pair_based_transforms = pair_based_transforms
        if same:
            if isinstance(self.n_elements, int):
                datasets_indices = [random.randrange(self.n_datasets)
                                    for _ in range(self.max_iters)]
                self.indices = [[(dataset_idx, data_idx)
                                 for data_idx in random.choices(range(self.n_data[dataset_idx]),
                                                                k=self.n_elements)]
                                for dataset_idx in datasets_indices]
            else:
                raise ValueError("When 'same=true', 'n_element' should be an integer.")
        else:
            if isinstance(self.n_elements, list):
                self.indices = [[(dataset_idx, data_idx)
                                 for i, dataset_idx in enumerate(
                                     random.sample(range(self.n_datasets), k=len(self.n_elements)))
                                 for data_idx in random.sample(range(self.n_data[dataset_idx]),
                                                               k=self.n_elements[i])]
                                for i_iter in range(self.max_iters)]
            elif self.n_elements > self.n_datasets:
                raise ValueError("When 'same=False', 'n_element' should be no more than n_datasets")
            else:
                self.indices = [[(dataset_idx, random.randrange(self.n_data[dataset_idx]))
                                 for dataset_idx in random.sample(range(self.n_datasets),
                                                                  k=n_elements)]
                                for i in range(max_iters)]

    def __len__(self):
        return self.max_iters

    def __getitem__(self, idx):
        sample = [self.datasets[dataset_idx][data_idx]
                  for dataset_idx, data_idx in self.indices[idx]]
        if self.pair_based_transforms is not None:
            for transform, args in self.pair_based_transforms:
                sample = transform(sample, **args)
        return sample




class Subset(Dataset):
    """
    Subset of a dataset at specified indices.

    Args:
        dataset:
            The whole Dataset
        indices:
            Indices in the whole set selected for subset
        sub_attrib_args:
            Subset-specific arguments for attribute functions, expected a dict
    """
    def __init__(self, dataset, indices, sub_attrib_args=None):
        self.dataset = dataset
        self.indices = indices
        self.sub_attrib_args = sub_attrib_args

    def __getitem__(self, idx):
        if self.sub_attrib_args is not None:
            for key in self.sub_attrib_args:
                # Make sure the dataset already has the corresponding attributes
                # Here we only make the arguments subset dependent
                #   (i.e. pass different arguments for each subset)
                self.dataset.aux_attrib_args[key].update(self.sub_attrib_args[key])
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

class VOC(BaseDataset):
    """
    Base Class for VOC Dataset

    Args:
        base_dir:
            VOC dataset directory
        split:
            which split to use
            choose from ('train', 'val', 'trainval', 'trainaug')
        transform:
            transformations to be performed on images/masks
        to_tensor:
            transformation to convert PIL Image to tensor
    """
    def __init__(self, base_dir, split, transforms=None, to_tensor=None):
        super(VOC, self).__init__(base_dir)
        self.split = split
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._label_dir = os.path.join(self._base_dir, 'SegmentationClassAug')
        self._inst_dir = os.path.join(self._base_dir, 'SegmentationObjectAug')
        self._scribble_dir = os.path.join(self._base_dir, 'ScribbleAugAuto')
        self._id_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')
        self.transforms = transforms
        self.to_tensor = to_tensor

        with open(os.path.join(self._id_dir, '{}.txt'.format(self.split)), 'r') as f:
            self.ids = f.read().splitlines()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Fetch data
        id_ = self.ids[idx]
        image = Image.open(os.path.join(self._image_dir, '{}.jpg'.format(id_)))
        semantic_mask = Image.open(os.path.join(self._label_dir, '{}.png'.format(id_)))
        instance_mask = Image.open(os.path.join(self._inst_dir, '{}.png'.format(id_)))
        scribble_mask = Image.open(os.path.join(self._scribble_dir, '{}.png'.format(id_)))
        sample = {'image': image,
                  'label': semantic_mask,
                  'inst': instance_mask,
                  'scribble': scribble_mask}

        # Image-level transformation
        if self.transforms is not None:
            sample = self.transforms(sample)
        # Save the original image (without normalization)
        image_t = torch.from_numpy(np.array(sample['image']).transpose(2, 0, 1))
        # Transform to tensor
        if self.to_tensor is not None:
            sample = self.to_tensor(sample)

        sample['id'] = id_
        sample['image_t'] = image_t

        # Add auxiliary attributes
        for key_prefix in self.aux_attrib:
            # Process the data sample, create new attributes and save them in a dictionary
            aux_attrib_val = self.aux_attrib[key_prefix](sample, **self.aux_attrib_args[key_prefix])
            for key_suffix in aux_attrib_val:
                # one function may create multiple attributes, so we need suffix to distinguish them
                sample[key_prefix + '_' + key_suffix] = aux_attrib_val[key_suffix]

        return sample



#
#
class COCOSeg(BaseDataset):
    """
    Modified Class for COCO Dataset

    Args:
        base_dir:
            COCO dataset directory
        split:
            which split to use (default is 2014 version)
            choose from ('train', 'val')
        transform:
            transformations to be performed on images/masks
        to_tensor:
            transformation to convert PIL Image to tensor
    """
    def __init__(self, base_dir, split, transforms=None, to_tensor=None):
        super(COCOSeg).__init__(base_dir)
        self.split = split + '2014'
        annFile = '{}/annotations/instances_{}.json'.format(base_dir, self.split)
        self.coco = COCO(annFile)

        self.ids = self.coco.getImgIds()
        self.transforms = transforms
        self.to_tensor = to_tensor

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Fetch meta data
        id_ = self.ids[idx]
        img_meta = self.coco.loadImgs(id_)[0]
        annIds = self.coco.getAnnIds(imgIds=img_meta['id'])

        # Open Image
        image = Image.open("{}/{}/{}".format(self._base_dir, self.split, img_meta['file_name']))
        if image.mode == 'L':
            image = image.convert('RGB')

        # Process masks
        anns = self.coco.loadAnns(annIds)
        semantic_masks = {}
        for ann in anns:
            catId = ann['category_id']
            mask = self.coco.annToMask(ann)
            if catId in semantic_masks:
                semantic_masks[catId][mask == 1] = catId
            else:
                semantic_mask = np.zeros((img_meta['height'], img_meta['width']), dtype='uint8')
                semantic_mask[mask == 1] = catId
                semantic_masks[catId] = semantic_mask
        semantic_masks = {catId: Image.fromarray(semantic_mask)
                          for catId, semantic_mask in semantic_masks.items()}

        # No scribble/instance mask
        instance_mask = Image.fromarray(np.zeros_like(semantic_mask, dtype='uint8'))
        scribble_mask = Image.fromarray(np.zeros_like(semantic_mask, dtype='uint8'))

        sample = {'image': image,
                  'label': semantic_masks,
                  'inst': instance_mask,
                  'scribble': scribble_mask}

        # Image-level transformation
        if self.transforms is not None:
            sample = self.transforms(sample)
        # Save the original image (without mean subtraction/normalization)
        image_t = torch.from_numpy(np.array(sample['image']).transpose(2, 0, 1))

        # Transform to tensor
        if self.to_tensor is not None:
            sample = self.to_tensor(sample)

        sample['id'] = id_
        sample['image_t'] = image_t

        # Add auxiliary attributes
        for key_prefix in self.aux_attrib:
            # Process the data sample, create new attributes and save them in a dictionary
            aux_attrib_val = self.aux_attrib[key_prefix](sample, **self.aux_attrib_args[key_prefix])
            for key_suffix in aux_attrib_val:
                # one function may create multiple attributes, so we need suffix to distinguish them
                sample[key_prefix + '_' + key_suffix] = aux_attrib_val[key_suffix]

        return sample





def attrib_basic(_sample, class_id):
    """
    Add basic attribute

    Args:
        _sample: data sample
        class_id: class label asscociated with the data
            (sometimes indicting from which subset the data are drawn)
    """
    return {'class_id': class_id}


def getMask(label, scribble, class_id, class_ids):
    """
    Generate FG/BG mask from the segmentation mask

    Args:
        label:
            semantic mask
        scribble:
            scribble mask
        class_id:
            semantic class of interest
        class_ids:
            all class id in this episode
    """
    # Dense Mask
    fg_mask = torch.where(label == class_id,
                          torch.ones_like(label), torch.zeros_like(label))
    bg_mask = torch.where(label != class_id,
                          torch.ones_like(label), torch.zeros_like(label))
    for class_id in class_ids:
        bg_mask[label == class_id] = 0

    # Scribble Mask
    bg_scribble = scribble == 0
    fg_scribble = torch.where((fg_mask == 1)
                              & (scribble != 0)
                              & (scribble != 255),
                              scribble, torch.zeros_like(fg_mask))
    scribble_cls_list = list(set(np.unique(fg_scribble)) - set([0,]))
    if scribble_cls_list:  # Still need investigation
        fg_scribble = fg_scribble == random.choice(scribble_cls_list).item()
    else:
        fg_scribble[:] = 0

    return {'fg_mask': fg_mask,
            'bg_mask': bg_mask,
            'fg_scribble': fg_scribble.long(),
            'bg_scribble': bg_scribble.long()}


def fewShot(paired_sample, n_ways, n_shots, cnt_query, coco=False):
    """
    Postprocess paired sample for fewshot settings

    Args:
        paired_sample:
            data sample from a PairedDataset
        n_ways:
            n-way few-shot learning
        n_shots:
            n-shot few-shot learning
        cnt_query:
            number of query images for each class in the support set
        coco:
            MS COCO dataset
    """
    ###### Compose the support and query image list ######
    cumsum_idx = np.cumsum([0,] + [n_shots + x for x in cnt_query])

    # support class ids
    class_ids = [paired_sample[cumsum_idx[i]]['basic_class_id'] for i in range(n_ways)]

    # support images
    support_images = [[paired_sample[cumsum_idx[i] + j]['image'] for j in range(n_shots)]
                      for i in range(n_ways)]
    support_images_t = [[paired_sample[cumsum_idx[i] + j]['image_t'] for j in range(n_shots)]
                        for i in range(n_ways)]

    # support image labels
    if coco:
        support_labels = [[paired_sample[cumsum_idx[i] + j]['label'][class_ids[i]]
                           for j in range(n_shots)] for i in range(n_ways)]
    else:
        support_labels = [[paired_sample[cumsum_idx[i] + j]['label'] for j in range(n_shots)]
                          for i in range(n_ways)]
    support_scribbles = [[paired_sample[cumsum_idx[i] + j]['scribble'] for j in range(n_shots)]
                         for i in range(n_ways)]
    support_insts = [[paired_sample[cumsum_idx[i] + j]['inst'] for j in range(n_shots)]
                     for i in range(n_ways)]



    # query images, masks and class indices
    query_images = [paired_sample[cumsum_idx[i+1] - j - 1]['image'] for i in range(n_ways)
                    for j in range(cnt_query[i])]
    query_images_t = [paired_sample[cumsum_idx[i+1] - j - 1]['image_t'] for i in range(n_ways)
                      for j in range(cnt_query[i])]
    if coco:
        query_labels = [paired_sample[cumsum_idx[i+1] - j - 1]['label'][class_ids[i]]
                        for i in range(n_ways) for j in range(cnt_query[i])]
    else:
        query_labels = [paired_sample[cumsum_idx[i+1] - j - 1]['label'] for i in range(n_ways)
                        for j in range(cnt_query[i])]
    query_cls_idx = [sorted([0,] + [class_ids.index(x) + 1
                                    for x in set(np.unique(query_label)) & set(class_ids)])
                     for query_label in query_labels]


    ###### Generate support image masks ######
    support_mask = [[getMask(support_labels[way][shot], support_scribbles[way][shot],
                             class_ids[way], class_ids)
                     for shot in range(n_shots)] for way in range(n_ways)]


    ###### Generate query label (class indices in one episode, i.e. the ground truth)######
    query_labels_tmp = [torch.zeros_like(x) for x in query_labels]
    for i, query_label_tmp in enumerate(query_labels_tmp):
        query_label_tmp[query_labels[i] == 255] = 255
        for j in range(n_ways):
            query_label_tmp[query_labels[i] == class_ids[j]] = j + 1

    ###### Generate query mask for each semantic class (including BG) ######
    # BG class
    query_masks = [[torch.where(query_label == 0,
                                torch.ones_like(query_label),
                                torch.zeros_like(query_label))[None, ...],]
                   for query_label in query_labels]
    # Other classes in query image
    for i, query_label in enumerate(query_labels):
        for idx in query_cls_idx[i][1:]:
            mask = torch.where(query_label == class_ids[idx - 1],
                               torch.ones_like(query_label),
                               torch.zeros_like(query_label))[None, ...]
            query_masks[i].append(mask)


    return {'class_ids': class_ids,

            'support_images_t': support_images_t,
            'support_images': support_images,
            'support_mask': support_mask,
            'support_inst': support_insts,

            'query_images_t': query_images_t,
            'query_images': query_images,
            'query_labels': query_labels_tmp,
            'query_masks': query_masks,
            'query_cls_idx': query_cls_idx,
           }


def voc_fewshot(base_dir, split, transforms, to_tensor, labels, n_ways, n_shots, max_iters,
                n_queries=1):
    """
    Args:
        base_dir:
            VOC dataset directory
        split:
            which split to use
            choose from ('train', 'val', 'trainval', 'trainaug')
        transform:
            transformations to be performed on images/masks
        to_tensor:
            transformation to convert PIL Image to tensor
        labels:
            object class labels of the data
        n_ways:
            n-way few-shot learning, should be no more than # of object class labels
        n_shots:
            n-shot few-shot learning
        max_iters:
            number of pairs
        n_queries:
            number of query images
    """
    voc = VOC(base_dir=base_dir, split=split, transforms=transforms, to_tensor=to_tensor)
    voc.add_attrib('basic', attrib_basic, {})

    # Load image ids for each class
    sub_ids = []
    for label in labels:
        with open(os.path.join(voc._id_dir, voc.split,
                               'class{}.txt'.format(label)), 'r') as f:
            sub_ids.append(f.read().splitlines())
    # Create sub-datasets and add class_id attribute
    subsets = voc.subsets(sub_ids, [{'basic': {'class_id': cls_id}} for cls_id in labels])

    # Choose the classes of queries
    # cnt_query = np.bincount(random.choices(population=range(n_ways), k=n_queries), minlength=n_ways)
    cnt_query = np.bincount(random.sample(population=range(n_ways), k=n_queries), minlength=n_ways)
    # Set the number of images for each class
    n_elements = [n_shots + x for x in cnt_query]
    # Create paired dataset
    paired_data = PairedDataset(subsets, n_elements=n_elements, max_iters=max_iters, same=False,
                                pair_based_transforms=[
                                    (fewShot, {'n_ways': n_ways, 'n_shots': n_shots,
                                               'cnt_query': cnt_query})])
    return paired_data


def coco_fewshot(base_dir, split, transforms, to_tensor, labels, n_ways, n_shots, max_iters,
                 n_queries=1):
    """
    Args:
        base_dir:
            COCO dataset directory
        split:
            which split to use
            choose from ('train', 'val')
        transform:
            transformations to be performed on images/masks
        to_tensor:
            transformation to convert PIL Image to tensor
        labels:
            labels of the data
        n_ways:
            n-way few-shot learning, should be no more than # of labels
        n_shots:
            n-shot few-shot learning
        max_iters:
            number of pairs
        n_queries:
            number of query images
    """
    cocoseg = COCOSeg(base_dir, split, transforms, to_tensor)
    cocoseg.add_attrib('basic', attrib_basic, {})

    # Load image ids for each class
    cat_ids = cocoseg.coco.getCatIds()
    sub_ids = [cocoseg.coco.getImgIds(catIds=cat_ids[i - 1]) for i in labels]
    # Create sub-datasets and add class_id attribute
    subsets = cocoseg.subsets(sub_ids, [{'basic': {'class_id': cat_ids[i - 1]}} for i in labels])

    # Choose the classes of queries
    cnt_query = np.bincount(random.choices(population=range(n_ways), k=n_queries),
                            minlength=n_ways)
    # Set the number of images for each class
    n_elements = [n_shots + x for x in cnt_query]
    # Create paired dataset
    paired_data = PairedDataset(subsets, n_elements=n_elements, max_iters=max_iters, same=False,
                                pair_based_transforms=[
                                    (fewShot, {'n_ways': n_ways, 'n_shots': n_shots,
                                               'cnt_query': cnt_query, 'coco': True})])
    return paired_data


if __name__ == '__main__':
    from torchvision.transforms import Compose
    from transforms.myTransforms import RandomMirror, Resize, ToTensorNormalize
    from utils import set_seed, CLASS_LABELS
    from torch.utils.data import DataLoader

    labels = CLASS_LABELS['VOC'][1]
    transforms = Compose([Resize(size=(417,417)),
                          RandomMirror()])
    dataset = voc_fewshot(
        base_dir='/home/liruimin/PANet-master/data/Pascal/VOCdevkit/VOC2012',
        split='trainaug',
        transforms=transforms,
        to_tensor=ToTensorNormalize(),
        labels=labels,
        max_iters= 1000 * 1,
        n_ways=1,
        n_shots=5,
        n_queries=1
    )
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1,pin_memory=True,drop_last=True)
    for idx, dat in enumerate(train_loader):
        print(idx)

    # print(dataset)