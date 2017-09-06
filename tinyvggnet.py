import torch
import torch.nn
from collections import OrderedDict

"""
Test net architecture.
"""

N_CLASSES = 200

class TinyVGGNet(torch.nn.Module):

    def __init__(self):
        super(TinyVGGNet, self).__init__()

        layers = OrderedDict()
        layers["stem"] = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1)
        layers["block_01"] = TinyVGGNet.__get_block(32, 32, 1)
        layers["block_02"] = TinyVGGNet.__get_block(32, 64, 2)
        layers["block_03"] = TinyVGGNet.__get_block(64, 64, 1)
        layers["block_04"] = TinyVGGNet.__get_block(64, 128, 2)
        layers["block_05"] = TinyVGGNet.__get_block(128, 128, 1)
        layers["block_06"] = TinyVGGNet.__get_block(128, 256, 2)
        layers["block_07"] = TinyVGGNet.__get_block(256, 256, 1)
        layers["block_08"] = TinyVGGNet.__get_block(256, 256, 2)
        layers["block_09"] = TinyVGGNet.__get_block(256, 256, 1)
        layers["pool"] = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.layers = layers
        self.vision_net = torch.nn.Sequential(layers)
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(256, N_CLASSES),
            torch.nn.Softmax())

    @staticmethod
    def __get_block(in_channels, out_channels, stride=1):
        block =  torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=stride),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.ReLU(inplace=True))
        return block

    def forward(self, x):
        assert len(x.size()) == 4
        u = self.vision_net(x)
        v = u.view(u.size()[0], u.size()[1])
        y = self.predictor(v)
        return y


if __name__ == "__main__":
    from pathlib import Path
    script_folder = str(Path(__file__).resolve().parents[0])

    BATCH_SIZE = 128
    IMAGE_SIZE = 56
    DEVICE_ID = 0

    model = TinyVGGNet()
    model.cuda(device_id=DEVICE_ID)

    loss_func = torch.nn.CrossEntropyLoss().cuda(device_id=DEVICE_ID)

    x = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    y = N_CLASSES * torch.rand(BATCH_SIZE) + 0.5
    y = y.long()
    x_var_cuda = torch.autograd.Variable(x.cuda(device=DEVICE_ID), requires_grad=True)
    y_var_cuda = torch.autograd.Variable(y.cuda(device=DEVICE_ID), requires_grad=False)

    print("Forward...")
    y_hat = model(x_var_cuda)

    loss_value = loss_func(y_hat, y_var_cuda)
    loss_value_float = (loss_value.data.cpu().numpy())[0]
    print("Loss value = {:<8.4f}".format(loss_value_float))

    print("Loss backward...")
    loss_value.backward()
    print('ShiftNet grad (input):', x_var_cuda.grad)

    print("Done succesfully.")
