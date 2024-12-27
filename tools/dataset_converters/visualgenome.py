import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert VisualGenome annotations to COCO format')
    parser.add_argument('visualgenome_path', help='visualgenome data path')
    parser.add_argument('--img-dir', default='leftImg8bit', type=str)
    parser.add_argument('--gt-dir', default='gtFine', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()


if __name__ == "__main__":
    main()