import torch
import torch.nn.functional as F

def print_tensor_preview(t, rows=5, cols=10, channels=5):
    """
    Print tensor shape and top-left preview for 4D tensors (N, C, H, W).
    Doesn't flood console.
    """
    if t.ndim != 4:
        raise ValueError("Expected a 4D tensor with shape (N, C, H, W).")
    
    N, C, H, W = t.shape
    print(f"Tensor shape: {t.shape}")

    # Limit preview
    r = min(rows, H)
    c = min(cols, W)
    nch = min(channels, C)

    # Print each channel separately
    for n in range(min(N, 1)):  # show only batch 0
        print(f"\n[Batch {n}]")
        for ch in range(nch):
            print(f" Channel {ch} (showing top-left {r}x{c}):")
            print(t[n, ch, :r, :c])
            print()


def upsample_data(flow, info, mask):
    """Upsample [H/8, W/8, C] -> [H, W, C] using convex combination"""
    N, C, H, W = info.shape
    mask = mask.view(N, 1, 9, 8, 8, H, W)
    mask = torch.softmax(mask, dim=2)

    up_flow = F.unfold(8 * flow, kernel_size=3, padding=1)
    up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
    up_info = F.unfold(info, kernel_size=3, padding=1)
    up_info = up_info.view(N, C, 9, 1, 1, H, W)

    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    up_info = torch.sum(mask * up_info, dim=2)
    up_info = up_info.permute(0, 1, 4, 2, 5, 3)

    return up_flow.reshape(N, 2, 8 * H, 8 * W), up_info.reshape(N, C, 8 * H, 8 * W)


def upsample_data2(flow, info, mask):
    """Upsample [H/8, W/8, C] -> [H, W, C] using convex combination"""
    N, C, H, W = info.shape

    mask = mask.view(N, 1, 9, 8, 8, H, W)
    mask = torch.softmax(mask, dim=2)
    
    up_flow = F.unfold(8 * flow, [3, 3], padding=1)
    up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
    up_info = F.unfold(info, [3, 3], padding=1)
    up_info = up_info.view(N, C, 9, 1, 1, H, W)

    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    up_info = torch.sum(mask * up_info, dim=2)
    up_info = up_info.permute(0, 1, 4, 2, 5, 3)

    return up_flow.reshape(N, 2, 8 * H, 8 * W), up_info.reshape(N, C, 8 * H, 8 * W)


def analyze_upsample_data(flow, info, mask):
    """Upsample [H/8, W/8, C] -> [H, W, C] using convex combination"""
    print("\n_____ANALYZE UPSAMPLE DATA_____")

    print(f"FLOW SHAPE: {flow.shape}")
    print(f"INFO SHAPE: {info.shape}")
    print(f"MASK SHAPE: {mask.shape}")
    
    N, C, H, W = info.shape

    mask = mask.view(N, 1, 9, 8, 8, H, W)
    mask = torch.softmax(mask, dim=2)

    up_flow = F.unfold(8 * flow, [3, 3], padding=1)
    up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
    up_info = F.unfold(info, [3, 3], padding=1)
    up_info = up_info.view(N, C, 9, 1, 1, H, W)

    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    up_info = torch.sum(mask * up_info, dim=2)
    up_info = up_info.permute(0, 1, 4, 2, 5, 3)

    up_flow = up_flow.reshape(N, 2, 8 * H, 8 * W)
    up_info = up_info.reshape(N, C, 8 * H, 8 * W)


def main():
    flow = torch.arange(1, 1*2*30*80 + 1).reshape(1, 2, 30, 80).float()
    # flow = torch.ones(1, 2, 30, 80)
    info = torch.rand(1, 4, 30, 80)
    mask = torch.rand(1, 576, 30, 80)

    analyze_upsample_data(flow, info, mask)

    flow_up, info_up = upsample_data(flow, info, mask)
    flow_up2, info_up2 = upsample_data2(flow, info, mask)

    # Check if the two results are the same
    print("\n_____Shape comparison_____")
    print(f"- Flow up: {flow_up.shape} == {flow_up2.shape}")
    print(f"- Info up: {info_up.shape} == {info_up2.shape}")

    print("\n_____Value comparison_____")
    print(f"- Flow up: {torch.allclose(flow_up, flow_up2)}")
    print(f"- Info up: {torch.allclose(info_up, info_up2)}")


if __name__ == "__main__":
    
    main()