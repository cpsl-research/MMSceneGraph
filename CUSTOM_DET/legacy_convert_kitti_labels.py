import os.path as osp
import mmcv
import glob


def convert_kitti_to_coco(label_folder, image_folder, out_file):
    """
    Converts kitti labels to coco format
    """
    obj_accept = ['Car', 'Pedestrian', 'Cyclist']
    obj_id_map = {o:i for i, o in enumerate(obj_accept)}

    annotations = []
    images = []
    obj_count = 0
    for label_file in sorted(glob.glob(osp.join(label_folder, '*.txt'))):
        idx_file = int(label_file.split('/')[-1].replace('.txt',''))
        img_fname = '%06d.png'%idx_file
        img_path = osp.join(image_folder, img_fname)
        assert osp.exists(img_path)

        height, width = 375, 1274
        images.append(dict(
            id=idx_file,
            file_name=img_fname,
            height=height,
            width=width))

        with open(label_file, 'r') as f:
            labels_raw = [o.strip().split() for o in f.readlines() if o.split()[0] in obj_accept]

        for lab in labels_raw:
            obj_type = lab[0]
            x_min, y_min, x_max, y_max = [int(float(c)) for c in lab[4:8]]
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
    print('Converting KITTI to COCO format...')
    folders = ['train', 'val']
    for fol in folders:
        print(f'...converting {fol}')
        kitti_base = f'/data/spencer/KITTI/object/{fol}_only/'
        kitti_label = osp.join(kitti_base, 'label_2')
        kitti_image = f'../data/KITTI/{fol}/image_2/data'
        out_file =    f'../data/KITTI/{fol}/image_2/annotation_kitti_in_coco.json'
        convert_kitti_to_coco(kitti_label, kitti_image, out_file)
        print('done')
