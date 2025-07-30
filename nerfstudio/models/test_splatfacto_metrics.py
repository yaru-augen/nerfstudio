import torch

from nerfstudio.models.splatfacto import SplatfactoModel

# Dummy outputs and batch for testing get_image_metrics_and_images
H, W = 4, 4
predicted_rgb = torch.rand(H, W, 3, device="cpu")
gt_img = torch.rand(H, W, 3, device="cpu")
background = torch.rand(3, device="cpu")
outputs = {"rgb": predicted_rgb, "background": background}
batch = {"image": (gt_img * 255).to(torch.uint8)}


def get_dummy_model():
    from nerfstudio.models.splatfacto import SplatfactoModelConfig

    config = SplatfactoModelConfig()
    model = SplatfactoModel(config=config)
    model.populate_modules()
    return model


if __name__ == "__main__":
    model = get_dummy_model()
    metrics, images = model.get_image_metrics_and_images(outputs, batch)
    print("Metrics:", metrics)
    print("Images dict keys:", images.keys())
    print("Images dict keys:", images.keys())
