import torch
import torch.nn as nn
import torch.nn.functional as F
from core.corr import CorrBlock1D, PytorchAlternateCorrBlock1D, CorrBlockFast1D, AlternateCorrBlock
from core.stereo import raftstereo
from core.semantic import RoadSeg_decoder
import torchvision
import os

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

def LRC(left_disparity, right_disparity, threshold):
    
    _, _, height, width = left_disparity.shape
    device = left_disparity.device
    left_disparity=left_disparity.squeeze()
    right_disparity=right_disparity.squeeze()

    x_grid = torch.arange(width, device=device).repeat(height, 1)
    
    x_right = x_grid - left_disparity
    valid_mask = (x_right >= 0) & (x_right < width)
    x_right[~valid_mask] = width-1
    disparity_right = torch.gather(right_disparity, 1, x_right.long())
    x_back_mapped = x_right + disparity_right
    # 计算一致性
    dismatrix = torch.abs(x_back_mapped - x_grid)
    dismatrix_exp=torch.sigmoid(dismatrix)
    Y = dismatrix_exp.unsqueeze(0).unsqueeze(0).float()
    return Y
    # #视差修正

class JLnet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.raft=raftstereo(self.args)
        self.seg_decoder = RoadSeg_decoder(num_labels=19, use_sne=False)

    def freeze_bn(self):
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def forward(self, image1, image2,image1_1,image2_1, threshold,iters=12,flow_init=None, test_mode=False):
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
        image1_1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2_1 = (2 * (image2 / 255.0) - 1.0).contiguous()
        if test_mode:
            displ,x1,x3,fmap1=self.raft(image1,image2,test_mode=test_mode)
            x1,x3,fmap1=x1.float(), x3.float(), fmap1.float()
            seg=self.seg_decoder(x1,x3,fmap1,displ,test_mode)
            return displ,seg
        else:
            flow_predictions,displ,x1,x3,fmap1=self.raft(image1,image2,test_mode=test_mode)
            _,              dispr,*ignore=self.raft(image1_1,image2_1,test_mode=test_mode)
            mask=LRC(displ,dispr,threshold)
            x1,x3,fmap1=x1.float(), x3.float(), fmap1.float()
            seg1,seg2,sge3,sge4,seg11,seg12,seg13=self.seg_decoder(x1,x3,fmap1,displ,test_mode)
            return flow_predictions,displ,mask,seg1,seg2,sge3,sge4,seg11,seg12,seg13
