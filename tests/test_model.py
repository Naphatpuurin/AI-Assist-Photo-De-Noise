import torch
from models.nafnet import NAFNet


def test_nafnet_forward_backward():
    model = NAFNet(
        img_channel=5,
        out_channel=4,
        width=16,  # lightweight config for fast test
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 1],
        dec_blk_nums=[1, 1, 1, 1],
    )
    # Batch size 2, 5 channels (4 RGGB + 1 ISO condition), 64x64
    x = torch.randn(2, 5, 64, 64, requires_grad=True)
    out = model(x)

    assert out.shape == (2, 4, 64, 64)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_model_parameter_count():
    model = NAFNet(
        img_channel=5,
        out_channel=4,
        width=32,
        middle_blk_num=2,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nNAFNet Total Parameters: {total_params / 1e6:.2f}M")
    assert total_params > 0
    assert total_params == trainable_params
