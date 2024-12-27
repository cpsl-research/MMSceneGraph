import os.path as osp
from tqdm import tqdm
import mmcv
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.geometry_utils import view_points


general_to_detection_class = \
    {'animal':'ignore',
     'human.pedestrian.personal_mobility':   'ignore',
     'human.pedestrian.stroller':            'ignore',
     'human.pedestrian.wheelchair':          'ignore',
     'movable_object.debris':                'ignore',
     'movable_object.pushable_pullable':     'ignore',
     'static_object.bicycle_rack':           'ignore',
     'vehicle.emergency.ambulance':          'ignore',
     'vehicle.emergency.police':             'ignore',
     'movable_object.barrier':               'barrier',
     'vehicle.bicycle':                      'bicycle',
     'vehicle.bus.bendy':                    'bus',
     'vehicle.bus.rigid':                    'bus',
     'vehicle.car':                          'car',
     'vehicle.construction':                 'construction_vehicle',
     'vehicle.motorcycle':                   'motorcycle',
     'human.pedestrian.adult':               'pedestrian',
     'human.pedestrian.child':               'pedestrian',
     'human.pedestrian.construction_worker': 'pedestrian',
     'human.pedestrian.police_officer':      'pedestrian',
     'movable_object.trafficcone':           'traffic_cone',
     'vehicle.trailer':                      'trailer',
     'vehicle.truck':                        'truck'}


def convert_nuscenes_to_coco(nusc, scene_splits, split_name, img_folder_base, out_file, cameras=['CAM_FRONT']):
    """
    Converts nuscenes labels to coco format

    Ability to select which cameras to use for training
    """
    if len(cameras) > 1 or cameras[0] != 'CAM_FRONT':
        raise NotImplementedError(cameras)
    obj_accept = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']
    obj_id_map = {o:i for i, o in enumerate(obj_accept)}

    annotations = []
    images = []
    obj_count = 0
    idx_file = 0

    # -- loop over scenes in this split
    for scene in tqdm(scene_splits):
        try:
            idx = int(scene.replace('scene-', ''))
            scene_data = nusc.scene[idx]
        except IndexError as e:
            print(f'Tried to index into {idx} from {scene}')
            continue
            # raise e

        # -- get all sample records ahead of time
        sample_records = {0:nusc.get('sample', scene_data['first_sample_token'])}
        for i in range(1, scene_data['nbr_samples'], 1):
            sample_records[i] = nusc.get('sample', sample_records[i-1]['next'])

        # -- loop over frames in the camera data
        for frame in range(scene_data['nbr_samples']):
            for cam in cameras:
                # -- image file path
                sensor_record = nusc.get('sample_data', sample_records[frame]['data'][cam])
                img_fname = sensor_record['filename']
                img_path = osp.join(img_folder_base, img_fname)
                assert osp.exists(img_path)

                # -- image information
                height, width = 900, 1600
                images.append(dict(
                    id=idx_file,
                    file_name=img_fname,
                    height=height,
                    width=width))
                idx_file += 1

                # -- label information
                _, boxes, cam_P = nusc.get_sample_data(sensor_record['token'])
                for box in boxes:
                    obj_type = general_to_detection_class[box.name]
                    if obj_type in obj_accept:
                        # -- project 3D box into 2D and convert projected 3D box to 2D box
                        corners_proj = view_points(box.corners(), cam_P, normalize=True)[:2,:]
                        ix = 0
                        iy = 1
                        x_min = min(corners_proj[ix,:])
                        x_max = max(corners_proj[ix,:])
                        y_min = min(corners_proj[iy,:])
                        y_max = max(corners_proj[iy,:])

                        # box coordinates are measured from top left and are 0-indexed
                        data_anno = dict(
                            image_id=idx_file,
                            id=obj_count,
                            category_id=obj_id_map[obj_type],
                            bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                            area=(x_max - x_min) * (y_max - y_min),
                            segmentation=[],
                            iscrowd=0)
                        annotations.append(data_anno)
                        obj_count += 1
    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{'id':i, 'name':n} for n, i in obj_id_map.items()])
    mmcv.dump(coco_format_json, out_file)


if __name__ == "__main__":
    data_dir = '/data/spencer/nuScenes'
    print('Loading nuscenes class...', end='', flush=True)
    nusc = NuScenes(version='v1.0-trainval', dataroot=data_dir, verbose=False)
    # nusc = NuScenes(version='v1.0-mini', dataroot=data_dir, verbose=False)

    print('done.')
    print('Creating scene splits...', end='', flush=True)
    scene_splits = create_splits_scenes()
    print('done.')
    img_folder_base = '../data/nuScenes/data/'

    # -- use "train" and "val" splits
    for split in ['train', 'val']:
        print(f'Converting {split}...')
        out_file = f'../data/nuScenes/{split}_annotation_nuscenes_in_coco.json'
        convert_nuscenes_to_coco(nusc, scene_splits[split], split, img_folder_base, out_file)
        print(f'done')
