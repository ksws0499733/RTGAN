import torch
import torch.nn as nn
from modeling.backbone.t2t_vit import T2T_ViT
from modeling.backbone.vit_mla import VIT_MLA
from modeling.backbone.xception import AlignedXception
import timm
def build_backbone(backbone, img_size =(512,512), 
                    in_chans = 3,
                    output_stride=16,
                    out_indices=[0,1,2,3,5], pretrain = True):
    print('  ------build_backbone------')
    if backbone.startswith('vit'):        
        depth = 12
        embed_dim = 256

        #vit-d12-e256
        parastr=backbone.split('-')[1:]
        for pstr in parastr:
            if pstr[0] == 'd':
                depth = int(pstr[1:])
            if pstr[0] == 'e':
                embed_dim = int(pstr[1:])

        print("    depth:",depth)
        print("    embed_dim",embed_dim)
        model = T2T_ViT( img_size=img_size, 
                        tokens_type='performer', 
                        in_chans=in_chans, 
                        embed_dim=embed_dim, 
                        num_heads=8,
                        token_dim=64,
                        depth=depth)
        output_para = {'layer_stride':[4,8,16,16], 
                       'layer_chan':[64,64,embed_dim,embed_dim]}
    elif backbone in ['mit_b0','mit_b1','mit_b2','mit_b3','mit_b4','mit_b5']:        
        from modeling.backbone.mit import mit_b0,mit_b1,mit_b2,mit_b3,mit_b4,mit_b5
        print("    backbone:",backbone)
        m = eval(backbone)  

        model = m(img_size=img_size,
                  in_chans=in_chans)


        i = torch.randn(1, in_chans, 64, 64)
        o = model(i)
        layer_strid = []
        layer_dim = []
        for x in o:
            layer_strid.append(64 // x.shape[2])
            layer_dim.append(x.shape[1])
        output_para = {'layer_stride':layer_strid, 
                       'layer_chan':layer_dim}                
    
    elif backbone == 'SETR':
        embed_dim = 1024
        norm_cfg = dict(type='BN', requires_grad=True)
        model = VIT_MLA( img_size=img_size, 
                        in_chans=in_chans,
                        embed_dim=embed_dim,
                        norm_cfg=norm_cfg)

        i = torch.randn(1, in_chans, 512, 512)
        o = model(i)
        layer_strid = []
        layer_dim = []
        for x in o:
            layer_strid.append(64 // x.shape[2])
            layer_dim.append(x.shape[1])
        output_para = {'layer_stride':layer_strid, 
                       'layer_chan':layer_dim} 
    
    elif backbone == 'xception':

        model = AlignedXception( output_stride=output_stride, 
                        BatchNorm=nn.BatchNorm2d)

        i = torch.randn(1, in_chans, 512, 512)
        o = model(i)
        layer_strid = []
        layer_dim = []
        for x in o:
            layer_strid.append(64 // x.shape[2])
            layer_dim.append(x.shape[1])
        output_para = {'layer_stride':layer_strid, 
                       'layer_chan':layer_dim}     
    else:
        model_names = timm.list_models(backbone,pretrained=False)
        if len(model_names) == 0:
            model_names = timm.list_models('*'+backbone+'*',pretrained=False)
            if len(model_names) > 0:
                raise RuntimeError('Unknown model, did you means {}'.format(model_names) )
            else:
                raise RuntimeError('Unknown model (%s)' % model_names)
        else:
            print('\t----[backbone]-----model_names', model_names)
            print('\t----[backbone]-----', backbone)
            
            model = timm.create_model(backbone,
                            pretrained=pretrain,
                            in_chans = in_chans,
                            features_only=True,
                            out_indices=out_indices,
                            output_stride=output_stride,
                            drop_rate=0.5,
                            drop_connect_rate=None,
                            drop_path_rate=0.1,
                            drop_block_rate=None,
                        )

            i = torch.randn(1, in_chans, 64, 64)
            o = model(i)
            layer_strid = []
            layer_dim = []
            for x in o:
                layer_strid.append(64 // x.shape[2])
                layer_dim.append(x.shape[1])


            # from thop import profile
            # input_test = torch.randn(1,3,512,512)
            # flops, params = profile(model, inputs=(input_test,))
            # model_name = backbone
            # print("\t----[backbone]-----  %s | %.3f | %.3f"%(model_name, params/(1000**2), flops/(1000**3)))
            

            output_para = {'layer_stride':layer_strid, 
                        'layer_chan':layer_dim}

    print('    layer_stride:',output_para['layer_stride'])
    print('    layer_channal:',output_para['layer_chan'])
    return model,output_para