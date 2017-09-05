import math
import numpy
import torch
import torch.nn
import torch.nn.functional

"""
This is the main building block of the ShiftNet.
It consists of two parts:
1) Function  that generates randomly shifted versions of input channels.
2) Normal 1x1 convolution that does dimension-reduction and generates output.
Basically it's equivalent to 3x3 depthwise conv + 1x1 conv.
No activation, no BatchNorm - add for your taste.
"""

class ShiftConv(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            stride=1,
            shifts_per_channel=8,
            receptive_field_radius=4.25,
            shifts_weighting="gaussian",
            shifts_weighting_params={"sigma": 1.6}):
        super(ShiftConv, self).__init__()
        self.__pad_size = int(receptive_field_radius)
        self.__zeropad = torch.nn.ZeroPad2d(self.__pad_size)
        self.__shifts = ShiftConv.__get_random_shifts(
            in_channels,
            shifts_per_channel,
            receptive_field_radius,
            self.__pad_size,
            shifts_weighting,
            shifts_weighting_params)
        self.__reduction = torch.nn.Conv2d(
            in_channels=in_channels * shifts_per_channel,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride)

    @staticmethod
    def __get_max_zeropad(receptive_field_radius):
        max_shift = int(receptive_field_radius)
        pad = torch.nn.ZeroPad2d((max_shift, max_shift, max_shift, max_shift))
        return pad

    @staticmethod
    def __get_gaussian_weighting(possible_shifts, sigma):
        p = []
        for dx, dy in possible_shifts:
            dist_sq = dx * dx + dy * dy
            w = math.exp(-dist_sq / (2.0 * sigma * sigma))
            p.append(w)
        p = numpy.array(p, dtype="float32")
        p /= numpy.sum(p)
        return p

    @staticmethod
    def __get_random_shifts(
            in_channels,
            shifts_per_channel,
            receptive_field_radius,
            pad_size,
            shifts_weighting,
            shifts_weighting_params):
        assert receptive_field_radius > 0
        epsilon = 0.000001
        possible_shifts = []
        for dx in range(-pad_size, pad_size+1):
            for dy in range(-pad_size, pad_size+1):
                dist = math.sqrt(dx * dx + dy * dy)
                if dist <= receptive_field_radius + epsilon:
                    possible_shifts.append((dx, dy))
        assert len(possible_shifts) >= shifts_per_channel
        weights = None
        if shifts_weighting == "gaussian":
            sigma = shifts_weighting_params["sigma"]
            weights = ShiftConv.__get_gaussian_weighting(possible_shifts, sigma)
        shifts = []
        indices = numpy.arange(len(possible_shifts), dtype="int")
        for channels_idx in range(in_channels):
            selected_indices = numpy.random.choice(indices, shifts_per_channel, replace=False, p=weights).tolist()
            for i in selected_indices:
                dx, dy = possible_shifts[i]
                shifts.append((channels_idx, dx, dy))
        return shifts

    @staticmethod
    def __apply_shifts(x, w, h, pad, shifts):
        accumulator = []
        for channel_idx, dx, dy in shifts:
            x1 = pad + dx
            x2 = x1 + w
            y1 = pad + dy
            y2 = y1 + h
            assert x1 >= 0
            assert x2 >= 0
            assert y1 >= 0
            assert x2 >= 0
            shifted_channel = x[:, channel_idx:(channel_idx+1), y1:y2, x1:x2]  # Notice dirty hack to keep array size.
            accumulator.append(shifted_channel)
        output = torch.cat(accumulator, 1)
        return output

    def forward(self, x):
        assert len(x.data.size()) == 4
        w = x.data.size()[3]
        h = x.data.size()[2]
        padding = self.__zeropad(x)
        expansion = ShiftConv.__apply_shifts(padding, w, h, self.__pad_size, self.__shifts)
        output = self.__reduction(expansion)
        return output


if __name__ == "__main__":
    import torch.autograd

    DEVICE_ID = 0
    IMAGE_SIZE = 224
    BATCH_SIZE = 16

    fake_x = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE).float()
    fake_x_cuda = fake_x.cuda(device=DEVICE_ID)
    fake_x_cuda_var = torch.autograd.Variable(fake_x_cuda, requires_grad=True)

    shiftconv = ShiftConv(3, 32)
    shiftconv.cuda(device_id=DEVICE_ID)

    # Fprop.
    print("Forward...")
    y = shiftconv(fake_x_cuda_var)

    # Bprop.
    print("Backward...")
    fake_grad = torch.randn(y.size()).float()
    fake_grad_cuda = fake_grad.cuda(device=DEVICE_ID)
    y.backward(fake_grad_cuda)

    print("Simple test done.")