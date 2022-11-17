
import os
import numpy as np
import cv2.cv2 as cv2

def saveimage(img,pred, root, subroot='0', startID=0, pred2 = None,gt=None, video_writer_list = None):

    pred_cls = np.argmax(pred, axis=1)      # pred mask
    # print(pred.shape)
    # print(pred_cls.shape)
    N,H,W,C = img.shape
    cls_color = [[0,0,0],[255,0,0],[0,255,0],[0,0,255],[255,0,255]]
    root = os.path.join(root,subroot)
    if not os.path.isdir(root):
        os.makedirs(root)
    for i in range(N):
        _img = img[i].astype('uint8')
        _prd = pred_cls[i].astype('float32')
        _heatmap = (pred[i]*255).astype('uint8')
        cls_mask = _prd*30
        im = np.zeros_like(_img)
        # im = Image.fromarray(cls_mask).convert('RGB')
        im_R = np.zeros_like(_prd)
        im_G = np.zeros_like(_prd)
        im_B = np.zeros_like(_prd)
        for j in range(5):
            im_R[_prd > j-0.1 ] = cls_color[j][0]
            im_G[_prd > j-0.1 ] = cls_color[j][1]
            im_B[_prd > j-0.1 ] = cls_color[j][2]
        im[:,:,0] = im_R
        im[:,:,1] = im_G
        im[:,:,2] = im_B
        imgout = cv2.addWeighted(im,0.5,_img,0.5,0)
        if video_writer_list is None:
            # npth = os.path.join(root,'{:0>6}.jpg'.format(startID+i))
            # npth2 = os.path.join(root,'{:0>6}_color.jpg'.format(startID+i))            
            # cv2.imwrite(npth,imgout)
            # cv2.imwrite(npth2,im)
            if gt is not None:
                _gt = gt[i].astype('float32')
                npth3 = os.path.join(root,'{:0>6}_gt.jpg'.format(startID+i))            
                # cv2.imwrite(npth3,_gt*255)

                im[:,:,1] = _gt*255
                imgcmp = cv2.addWeighted(im,0.35,_img,0.65,0)
                npth4 = os.path.join(root,'{:0>6}_gt_cmp.jpg'.format(startID+i))            
                cv2.imwrite(npth4,imgcmp)
                # print(_heatmap.shape)
                imgheat = cv2.applyColorMap(_heatmap[0],2)
                npth5 = os.path.join(root,'{:0>6}_heat.jpg'.format(startID+i))            
                cv2.imwrite(npth5,imgheat)                
        else:
            video_writer_list[0].write(imgout)
