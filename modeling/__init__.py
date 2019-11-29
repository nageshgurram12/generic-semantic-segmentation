from modeling.deeplab import *
from modeling.EMANet import *
from modeling.MFNet import *

def build_model(args, **kwargs):
    if args.model == 'Deeplab':
        model = DeepLab(num_classes=kwargs['nclass'],
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)
        
        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
    elif args.model == 'EMANet':
        if args.backbone == 'resnet':
            nlayers = 50
        else:
            raise NotImplementedError
            
        model = EMANet(kwargs['nclass'], nlayers, sync_bn=args.sync_bn,
                       stride=args.out_stride, mom=args.momentum)
        train_params=[
                {
                    'params': model.get_params(key='1x'),
                    'lr': 1 * args.lr,
                    'weight_decay': args.weight_decay,
                },
                {
                    'params': model.get_params(key='1y'),
                    'lr': 1 * args.lr,
                    'weight_decay': 0,
                },
                {
                    'params': model.get_params(key='2x'),
                    'lr': 2 * args.lr,
                    'weight_decay': 0.0,
                }]
        
    elif args.model == 'MFNet':
        model = MFNet(num_classes=kwargs['nclass'],
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)
        
        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
    
    return model, train_params