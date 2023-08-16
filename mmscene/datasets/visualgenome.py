# Copyright (c) OpenMMLab. All rights reserved.

from typing import List
import json

from mmscene.registry import DATASETS
from .base_det_dataset import BaseDetDataset


@DATASETS.register_module()
class VisualGenomeDataset(BaseDetDataset):
    """Dataset for VisualGenome."""

    METAINFO = {
        'classes': 
            ('__background__', 'airplane', 'animal', 'arm', 'bag', 'banana',
             'basket', 'beach', 'bear', 'bed', 'bench', 'bike', 'bird', 'board',
             'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch',
             'building', 'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child',
             'clock', 'coat', 'counter', 'cow', 'cup', 'curtain', 'desk', 'dog',
             'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
             'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl',
             'glass', 'glove', 'guy', 'hair', 'hand', 'handle', 'hat', 'head', 'helmet',
             'hill', 'horse', 'house', 'jacket', 'jean', 'kid', 'kite', 'lady', 'lamp',
             'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men', 'motorcycle',
             'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper',
             'paw', 'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant',
             'plate', 'player', 'pole', 'post', 'pot', 'racket', 'railing', 'rock',
             'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt', 'shoe',
             'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier',
             'sneaker', 'snow', 'sock', 'stand', 'street', 'surfboard', 'table',
             'tail', 'tie', 'tile', 'tire', 'toilet', 'towel', 'tower', 'track',
             'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable',
             'vehicle', 'wave', 'wheel', 'window', 'windshield', 'wing', 'wire',
             'woman', 'zebra'),
        'relations':
            ('__background__', 'above', 'across', 'against', 'along', 'and', 'at',
             'attached to', 'behind', 'belonging to', 'between', 'carrying', 'covered in',
             'covering', 'eating', 'flying in', 'for', 'from', 'growing on', 'hanging from',
             'has', 'holding', 'in', 'in front of', 'laying on', 'looking at', 'lying on',
             'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over', 'painted on',
             'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
             'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing',
             'wears', 'with'),
        'attributes':
            ('__background__', 'white', 'black', 'blue', 'green', 'red', 'brown', 'yellow',
             'small', 'large', 'wooden', 'silver', 'orange', 'grey', 'tall', 'long', 'dark',
             'pink', 'standing', 'round', 'tan', 'glass', 'here', 'wood', 'open', 'purple',
             'short', 'plastic', 'parked', 'sitting', 'walking', 'striped', 'brick', 'young',
             'gold', 'old', 'hanging', 'empty', 'on', 'bright', 'concrete', 'cloudy',
             'colorful', 'one', 'beige', 'bare', 'wet', 'light', 'square', 'closed', 'stone',
             'shiny', 'thin', 'dirty', 'flying', 'smiling', 'painted', 'thick', 'part',
             'sliced', 'playing', 'tennis', 'calm', 'leather', 'distant', 'rectangular',
             'looking', 'grassy', 'dry', 'cement', 'leafy', 'wearing', 'tiled', "man's",
             'baseball', 'cooked', 'pictured', 'curved', 'decorative', 'dead', 'eating',
             'paper', 'paved', 'fluffy', 'lit', 'back', 'framed', 'plaid', 'dirt', 'watching',
             'colored', 'stuffed', 'clean', 'in the picture', 'steel', 'stacked', 'covered',
             'full', 'three', 'street', 'flat', 'baby', 'black and white', 'beautiful',
             'ceramic', 'present', 'grazing', 'sandy', 'golden', 'blurry', 'side', 'chocolate',
             'wide', 'growing', 'chrome', 'cut', 'bent', 'train', 'holding', 'water', 'up',
             'arched', 'metallic', 'spotted', 'folded', 'electrical', 'pointy', 'running',
             'leafless', 'electric', 'in background', 'rusty', 'furry', 'traffic', 'ripe',
             'behind', 'laying', 'rocky', 'tiny', 'down', 'fresh', 'floral', 'stainless steel',
             'high', 'surfing', 'close', 'off', 'leaning', 'moving', 'multicolored', "woman's",
             'pair', 'huge', 'some', 'background', 'chain link', 'checkered', 'top', 'tree',
             'broken', 'maroon', 'iron', 'worn', 'patterned', 'ski', 'overcast', 'waiting',
             'rubber', 'riding', 'skinny', 'grass', 'porcelain', 'adult', 'wire', 'cloudless',
             'curly', 'cardboard', 'jumping', 'tile', 'pointed', 'blond', 'cream', 'four', 'male',
             'smooth', 'hazy', 'computer', 'older', 'pine', 'raised', 'many', 'bald', 'snow covered',
             'skateboarding', 'narrow', 'reflective', 'rear', 'khaki', 'extended', 'roman', 'american')
    }

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # obtain images that contain annotation
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            all_is_crowd = all([
                instance['ignore_flag'] == 1
                for instance in data_info['instances']
            ])
            if filter_empty_gt and (img_id not in ids_in_cat or all_is_crowd):
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos


def load_info(dict_file, add_bg=True):
    """
    Loads the file containing the visual genome label meanings
    """
    info = json.load(open(dict_file, 'r'))
    if add_bg:
        info['label_to_idx']['__background__'] = 0
        info['predicate_to_idx']['__background__'] = 0
        info['attribute_to_idx']['__background__'] = 0

    class_to_ind = info['label_to_idx']
    predicate_to_ind = info['predicate_to_idx']
    attribute_to_ind = info['attribute_to_idx']
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
    ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])
    ind_to_attributes = sorted(attribute_to_ind, key=lambda k: attribute_to_ind[k])

    return ind_to_classes, ind_to_predicates, ind_to_attributes