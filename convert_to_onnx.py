import torch
import torch.onnx
import argparse
import numpy as np

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
        input1 = torch.rand((1, 3, HEIGHT, WIDTH)).cuda()
        input2 = torch.rand((1, 3, HEIGHT, WIDTH)).cuda()

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
