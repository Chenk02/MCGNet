from thop import profile
import torch
from ptflops import get_model_complexity_info
from models.hdnet import HDNet

from sunrgbd.model_util_sunrgbd import SunrgbdDatasetConfig
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATASET_CONFIG = SunrgbdDatasetConfig()
end_points = {}
net = HDNet(num_class=DATASET_CONFIG.num_class,
           num_heading_bin=DATASET_CONFIG.num_heading_bin,
           num_size_cluster=DATASET_CONFIG.num_size_cluster,
           mean_size_arr=DATASET_CONFIG.mean_size_arr,
           num_proposal=256,
           input_feature_dim=1,
           vote_factor=1,
           sampling="vote_fps",
           with_angle=True)

net.to(device)

# checkpoint = torch.load("log_sunrgbd/checkpoint_eval259.tar")
# checkpoint_multigpu = dict()

# net.load_state_dict(checkpoint['model_state_dict'])

input = torch.randn(1, 40000, 4).to(device)  # 模型输入的形状,batch_size=1
inputs = {'point_clouds': input}
flops, params = profile(net, inputs=(inputs,end_points))
print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
