import torch
import torch.onnx
import argparse
import numpy as np
from PIL import Image
from models.recover import ReCoVEr_MN, ReCoVEr_RN, ReCoVEr_CX


class ModelWrapper(torch.nn.Module):
    """Wrapper class to export model with test_mode=True"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, image1, image2):
        # preprocess(image1, image2)
        output = self.model(image1, image2, test_mode=True, disable_cost=True)
        return output["final"]

def read_img(file):
    img = np.array(Image.open(str(file)))
    img = torch.Tensor(img)
    
    # Handle grayscale images (2D: H, W) by converting to RGB (3D: H, W, 3)
    if img.dim() == 2:
        img = img.unsqueeze(2)  # Add channel dimension: (H, W) -> (H, W, 1)
        img = img.repeat(1, 1, 3)  # Duplicate to RGB: (H, W, 1) -> (H, W, 3)
    # Handle RGBA images (4 channels) by converting to RGB
    elif img.dim() == 3 and img.size(2) == 4:
        img = img[:, :, :3]  # Take only RGB channels, drop alpha
    
    img = img.permute(2, 0, 1)  # HWC -> CHW

    return img


if __name__ == "__main__":
    WIDTH = 640
    HEIGHT = 240

    model = ReCoVEr_MN(pretrained=True)
    model = model.cuda()
    model.eval()

    # Wrap the model to handle test_mode and output extraction
    wrapped_model = ModelWrapper(model).cuda()
    wrapped_model.eval()

    with torch.no_grad():


        input1 = read_img("/hdd/data/projected_images/gray/10kph_2/0/000600.png").unsqueeze(0).cuda()
        input2 = read_img("/hdd/data/projected_images/gray/10kph_2/0/000606.png").unsqueeze(0).cuda()

        # input1 = torch.rand((1, 3, HEIGHT, WIDTH)).cuda()
        # input2 = torch.rand((1, 3, HEIGHT, WIDTH)).cuda()

        # Export to ONNX
        torch.onnx.export(
            wrapped_model,
            (input1, input2),
            "recover_mn.onnx",
            input_names=["image1", "image2"],
            output_names=["output"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes={}
        )
        print(f"Model exported to recover_mn.onnx")
