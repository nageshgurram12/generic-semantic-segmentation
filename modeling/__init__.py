from modeling.deeplab import *
from modeling.EMANet import *

def build_model(args, **kwargs):
    if args.model == 'Deeplab':
        model = DeepLab(num_classes=kwargs['nclass'],
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)
    elif args.model == 'EMANet':
        if args.backbone == 'resnet':
            nlayers = 50
        else:
            raise NotImplementedError
            
        model = EMANet(kwargs['nclass'], nlayers, sync_bn=args.sync_bn,
                       stride=args.out_stride)
        
    return model