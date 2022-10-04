import os.path as osp
import os
import json
from tqdm import tqdm
import glob
import argparse
import logging
import cv2

import avstack
import avapi

from avstack.objects import Occlusion


def convert_avstack_to_coco(SM, scene_splits, out_file, cameras=['CAM_FRONT']):
    """
    Converts avstack labels to coco format

    Ability to select which cameras to use for traning

    INPUTS:
    SM -- scene manager
    """
    obj_accept = SM.nominal_whitelist_types
    obj_id_map = {o:i for i, o in enumerate(obj_accept)}

    annotations = []
    images = []
    obj_count = 0
    idx_file = 0
    first = True

    # -- loop over scenes in this split
    n_problems = 0
    n_ignored = 0
    for scene in tqdm(scene_splits):
        try:
            SD = SM.get_scene_dataset_by_name(scene)
        except IndexError as e:
            logging.exception(e)
            print(f'Could not process scene {scene}...continuing')
            continue
        # print(f'...running scene {scene}')
        for cam in cameras:
            frames = SD.get_frames(sensor=cam)
            height, width = None, None
            for idx in range(0, len(frames), args.n_skips+1):
                i_frame = frames[idx]
                if i_frame < 3:  # don't do the first few frames
                    continue
                # -- image information
                # img = SD.get_image(i_frame, sensor=cam)
                calib = SD.get_calibration(i_frame, sensor=cam)
                try:
                    objs = SD.get_objects(i_frame, sensor=cam)
                    if len(objs) == 0:
                        n_ignored += 1
                        continue
                except KeyError as e:
                    n_problems += 1
                    logging.exception(e)
                    continue
                if height is None:
                    height = int(calib.height)
                    width = int(calib.width)
                img_filepath = SD.get_sensor_data_filepath(i_frame, sensor=cam)
                if not os.path.exists(img_filepath):
                    logging.warning(f'Could not find image filepath at {img_filepath}')
                    n_problems += 1
                    continue
                else:
                    try:
                        img = cv2.imread(img_filepath)
                        if img is None:
                            raise
                    except Exception as e:
                        n_problems += 1
                        logging.warning(f'Problem reading image at {img_filepath}')
                        continue
                images.append(dict(
                    id=idx_file,
                    file_name=img_filepath,
                    height=height,
                    width=width))
                idx_file += 1

                # -- object information
                for obj in objs:
                    if obj.occlusion == Occlusion.UNKNOWN:
                        try:
                            d_img = SD.get_depthimage(i_frame, cam + '_DEPTH')
                        except KeyError as e:
                            if first:
                                logging.exception(e)
                            occ = obj.occlusion
                            first = False
                    else:
                        occ = obj.occlusion

                    # -- filter based on occlusion
                    if obj.occlusion in [Occlusion.NONE, Occlusion.PARTIAL, \
                                         Occlusion.MOST, Occlusion.UNKNOWN]:
                        bbox_2d = obj.box.project_to_2d_bbox(calib)
                    else:
                        continue

                    # -- box coordinates are measured from top left and are 0-indexed
                    x_min, y_min, x_max, y_max = bbox_2d.box2d
                    x_min = int(x_min); y_min = int(y_min); x_max = int(x_max); y_max = int(y_max)
                    data_anno = dict(
                        image_id=idx_file,
                        id=obj_count,
                        category_id=obj_id_map[obj.obj_type],
                        bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                        area=(x_max - x_min) * (y_max - y_min),
                        segmentation=[],
                        iscrowd=0)
                    annotations.append(data_anno)
                    obj_count += 1

    # -- store annotations
    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{'id':i, 'name':n} for n, i in obj_id_map.items()])
    json.dump(coco_format_json, open(out_file, 'w'))
    print(f'{idx_file} valid images; {n_problems} problems; {n_ignored} ignored with this set')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Wrap avstack data to coco format for training')
    parser.add_argument('dataset', choices=['carla', 'kitti', 'nuscenes'], help='Choice of dataset')
    parser.add_argument('data_dir', type=str, help='Path to main dataset storage location')
    parser.add_argument('--n_skips', default=0, type=int, help='Number of skips between frames of a sequence')
    args = parser.parse_args()

    # -- create scene manager and get scene splits
    if args.dataset == 'carla':
        SM = avapi.carla.CarlaSceneManager(args.data_dir)
        cameras = ['CAM_FRONT', 'CAM_BACK', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
        splits_scenes = avapi.carla.get_splits_scenes(args.data_dir)
    elif args.dataset == 'kitti':
        cameras = ['image-2']
        splits_scenes = avapi.kitti.splits_scenes
        raise
    elif args.dataset == 'nuscenes':
        SM = avapi.nuscenes.nuScenesManager(args.data_dir, split="v1.0-trainval")
        cameras = ['CAM_FRONT', 'CAM_BACK', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        splits_scenes = avapi.nuscenes.splits_scenes
    else:
        raise NotImplementedError(args.dataset)

    # -- run main call
    for split in ['train', 'val']:
        print(f'Converting {split}...')
        out_file = f'../data/{args.dataset}/{split}_annotation_{args.dataset}_in_coco.json'
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        convert_avstack_to_coco(SM, splits_scenes[split], out_file, cameras)
        print(f'done')
