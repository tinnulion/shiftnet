import time
import torch
import torch.nn
import torch.autograd

DEVICE_ID = 0


def test_conv(batch_size, image_size, in_channels, out_channels, kernel_size, n_tests):
    print("Testing ordinary CONV with:")
    print("  Batch size   = {:d}".format(batch_size))
    print("  Image size   = {:d} x {:d}".format(image_size, image_size))
    print("  Channels     = {:d} -> {:d}".format(in_channels, out_channels))
    print("  Kernel size  = {:d} x {:d}".format(kernel_size, kernel_size))

    pad = kernel_size // 2
    normalconv = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=pad),
        torch.nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0))
    normalconv.cuda(device_id=DEVICE_ID)

    x = torch.randn(batch_size, in_channels, image_size, image_size).float()
    x_cuda = x.cuda(device=DEVICE_ID)
    x_cuda_var = torch.autograd.Variable(x_cuda, volatile=True)
    start = time.time()
    for _ in range(n_tests):
        y = normalconv(x_cuda_var)
    total_duration_inference = 1000.0 * (time.time() - start) / n_tests

    x = torch.randn(batch_size, in_channels, image_size, image_size).float()
    x_cuda = x.cuda(device=DEVICE_ID)
    x_cuda_var = torch.autograd.Variable(x_cuda, requires_grad=True)
    fake_grad = torch.randn(batch_size, out_channels, image_size, image_size).float()
    fake_grad_cuda = fake_grad.cuda(device=DEVICE_ID)
    start = time.time()
    for _ in range(n_tests):
        y = normalconv(x_cuda_var)
        y.backward(fake_grad_cuda)
    total_duration_train = 1000.0 * (time.time() - start) / n_tests

    print("Inference time = {:.2f} ms per batch".format(total_duration_inference))
    print("Train time     = {:.2f} ms per batch".format(total_duration_train))
    print("----")


def test_shiftconv(batch_size, image_size, in_channels, out_channels, shifts_per_channel, n_tests):
    import shiftconv

    print("Testing SHIFTCONV with:")
    print("  Batch size   = {:d}".format(batch_size))
    print(" Image size   = {:d} x {:d}".format(image_size, image_size))
    print("  Channels     = {:d} -> {:d}".format(in_channels, out_channels))
    print("  Num shifts   = {:d}".format(shifts_per_channel))

    shiftconv = shiftconv.ShiftConv(in_channels, out_channels)
    shiftconv.cuda(device_id=DEVICE_ID)

    x = torch.randn(batch_size, in_channels, image_size, image_size).float()
    x_cuda = x.cuda(device=DEVICE_ID)
    x_cuda_var = torch.autograd.Variable(x_cuda, volatile=True)
    start = time.time()
    for _ in range(n_tests):
        y = shiftconv(x_cuda_var)
    total_duration_inference = 1000.0 * (time.time() - start) / n_tests

    x = torch.randn(batch_size, in_channels, image_size, image_size).float()
    x_cuda = x.cuda(device=DEVICE_ID)
    x_cuda_var = torch.autograd.Variable(x_cuda, requires_grad=True)
    fake_grad = torch.randn(batch_size, out_channels, image_size, image_size).float()
    fake_grad_cuda = fake_grad.cuda(device=DEVICE_ID)
    start = time.time()
    for _ in range(n_tests):
        y = shiftconv(x_cuda_var)
        y.backward(fake_grad_cuda)
    total_duration_train = 1000.0 * (time.time() - start) / n_tests

    print("Inference time = {:.2f} ms per batch".format(total_duration_inference))
    print("Train time     = {:.2f} ms per batch".format(total_duration_train))
    print("----")


if __name__ == "__main__":
    from pathlib import Path
    script_folder = str(Path(__file__).resolve().parents[0])

    # Warm up.
    x = torch.randn(16, 3, 224, 224).float()
    x_cuda = x.cuda(device=DEVICE_ID)
    x_cuda_var = torch.autograd.Variable(x_cuda, volatile=True)
    conv = torch.nn.Conv2d(3, 32, kernel_size=3)
    conv.cuda(device_id=DEVICE_ID)
    y = conv(x_cuda_var)

    test_conv(16, 224, 3, 32, 3, 1000)
    test_conv(16, 224, 3, 32, 5, 1000)
    test_conv(16, 224, 3, 32, 7, 1000)

    print()

    test_conv(16, 224, 32, 32, 3, 1000)
    test_conv(16, 224, 32, 32, 5, 1000)
    test_conv(16, 224, 32, 32, 7, 1000)

    print()

    test_shiftconv(16, 224, 3, 32, 2, 1000)
    test_shiftconv(16, 224, 3, 32, 4, 1000)
    test_shiftconv(16, 224, 3, 32, 8, 1000)

    print()

    test_shiftconv(16, 224, 32, 32, 2, 1000)
    test_shiftconv(16, 224, 32, 32, 4, 1000)
    test_shiftconv(16, 224, 32, 32, 8, 1000)

    print()
    print("All done.")