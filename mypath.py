import os

class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return os.getenv('PASCAL_DATASET_PATH', '../datasets/pascal-voc-2012/VOCdevkit/VOC2012/')  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return os.getenv('SBD_DATASET_PATH','/path/to/datasets/benchmark_RELEASE/')  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return os.getenv('CITYSCAPES_DATASET_PATH', '/path/to/datasets/cityscapes/')     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return os.getenv('COCO_DATASET_PATH','../coco-data')
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
