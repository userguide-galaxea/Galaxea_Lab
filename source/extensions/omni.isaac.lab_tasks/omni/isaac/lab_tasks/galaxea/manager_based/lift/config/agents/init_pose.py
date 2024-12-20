import torch


# left_ee_pos:  tensor([[[0.3864, 0.5237, 1.1475]]], device='cuda:0')
# left_ee_quat:  tensor([[[ 9.6247e-05,  9.7698e-01, -2.1335e-01,  3.9177e-04]]],
#        device='cuda:0')
# right_ee_pos:  tensor([[[ 0.3864, -0.5237,  1.1475]]], device='cuda:0')
# right_ee_quat:  tensor([[[ 9.4220e-05, -9.7698e-01, -2.1335e-01, -3.8892e-04]]],
#        device='cuda:0')
LEFT_EE_POSE = torch.tensor(
    [0.3864, 0.5237, 1.1475, 9.6247e-05, 9.7698e-01, -2.1335e-01, 3.9177e-04],
    device="cuda:0",
)
RIGHT_EE_POSE = torch.tensor(
    [0.3864, -0.5237, 1.1475, 9.4220e-05, -9.7698e-01, -2.1335e-01, -3.8892e-04],
    device="cuda:0",
)
