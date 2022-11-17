

from logging import raiseExceptions


def build_generator(model_names,backbone = None,neck='LSN',output_stride=16,num_classes=2,in_chans=3,pretrain=False):
    print('\n------build_generator------\n')

    if model_names.startswith('vit') and neck == 'LSN':
        from .Vit_LSN import Vit_LSN as m
        print('  model:',model_names)
        generator_model = m(backbone=model_names,
                            output_stride = output_stride , 
                            num_classes=num_classes, 
                            in_chans=in_chans,
                            pretrain=pretrain)
    elif model_names.startswith('vit') and neck == 'AN':
        from .Vit_Deeplab import Vit_Deeplab as m
        generator_model = m( backbone=model_names,
                            output_stride = output_stride , 
                            num_classes=num_classes, 
                            in_chans=in_chans,
                            pretrain=pretrain)
    elif model_names.startswith('mit'):
        from .segFromer import SegFormer as m
        generator_model = m( backbone=model_names,
                            output_stride = output_stride , 
                            num_classes=num_classes, 
                            in_chans=in_chans,
                            pretrain=pretrain)
     
    
    elif model_names == 'csp' and neck == 'LSN':
        from .Csp_LSN import Csp_LSN as m
        generator_model = m( backbone=backbone,
                                output_stride = output_stride , 
                                num_classes=num_classes, 
                                in_chans=in_chans,
                                pretrain=pretrain)
    elif model_names == 'csp' and neck == 'LSN2':
        from .Csp_LSN2 import Csp_LSN as m
        generator_model = m( backbone=backbone,
                                output_stride = output_stride , 
                                num_classes=num_classes, 
                                in_chans=in_chans,
                                pretrain=pretrain)                               
    elif model_names == 'csp' and neck == 'AN':
        from .Csp_Deeplab import Csp_Deeplab as m
        generator_model = m( backbone=backbone,
                                output_stride = output_stride , 
                                num_classes=num_classes, 
                                in_chans=in_chans,
                                pretrain=pretrain)
    elif model_names == 'mobile'and neck == 'LSN':
        from .Mobile_LSN import Mobile_LSN as m
        generator_model = m( backbone=backbone,
                                output_stride = output_stride , 
                                num_classes=num_classes, 
                                in_chans=in_chans,
                                pretrain=pretrain)
    elif model_names == 'mobile' and neck == 'AN':
        from .Mobile_Deeplab import Mobile_Deeplab as m
        generator_model = m( backbone=backbone,
                                output_stride = output_stride , 
                                num_classes=num_classes, 
                                in_chans=in_chans,
                                pretrain=pretrain)
    
    elif model_names == 'segNet':
        from .SegNet import SegNet
        generator_model = SegNet(
                                input_channels=in_chans,
                                output_channels=num_classes,                                 
                                pretrain=True)

    elif model_names == 'UNet':
        from .UNet import UNET
        generator_model = UNET(
                                in_channel=in_chans,
                                out_channel=num_classes)

    elif model_names == 'pspNet':
        from .PSPNet import Pspnet as m    
        generator_model = m( backbone=backbone,
                                output_stride = output_stride , 
                                num_classes=num_classes, 
                                in_chans=in_chans,
                                pretrain=pretrain)   
    elif model_names == 'SETR' and neck == 'MLA':
        from .SETR_MLA import SETR_MLA as m  
        generator_model = m( backbone='SETR',
                                output_stride = output_stride , 
                                num_classes=num_classes, 
                                in_chans=in_chans,
                                pretrain=pretrain)    
    elif model_names == 'SETR' and neck =='UP':
        from .SETR_PUP import SETR_PUP as m  
        generator_model = m( backbone='SETR',
                                output_stride = output_stride , 
                                num_classes=num_classes, 
                                in_chans=in_chans,
                                pretrain=pretrain)  
    elif model_names == 'Deeplab':
        from .deeplabV3p import Deeplab as m  
        generator_model = m( backbone='xception',
                                output_stride = output_stride , 
                                num_classes=num_classes, 
                                in_chans=in_chans,
                                pretrain=False)  
    else:
        raise RuntimeError('Unknown model (%s)' % model_names)
    return generator_model