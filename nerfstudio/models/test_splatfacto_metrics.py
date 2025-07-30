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
    import torch

    from nerfstudio.data.scene_box import OrientedBox
    from nerfstudio.models.splatfacto import SplatfactoModelConfig

    config = SplatfactoModelConfig()
    # Create a dummy OrientedBox (scene_box) and set num_train_data=1
    scene_box = OrientedBox(torch.zeros(3), torch.ones(3), torch.eye(3))
    num_train_data = 1
    model = SplatfactoModel(
        config=config, scene_box=scene_box, num_train_data=num_train_data
    )
    model.populate_modules()
    return model


if __name__ == "__main__":
    model = get_dummy_model()
    # Print raw shapes and dtypes for debugging
    print(
        "raw batch['image'] shape:",
        batch["image"].shape,
        "dtype:",
        batch["image"].dtype,
    )
    print(
        "raw outputs['rgb'] shape:",
        outputs["rgb"].shape,
        "dtype:",
        outputs["rgb"].dtype,
    )
    gt_rgb = model.composite_with_background(
        model.get_gt_img(batch["image"]), outputs["background"]
    )
    predicted_rgb = outputs["rgb"]
    print("gt_rgb shape:", gt_rgb.shape, "dtype:", gt_rgb.dtype)
    print("predicted_rgb shape:", predicted_rgb.shape, "dtype:", predicted_rgb.dtype)
    metrics, images = model.get_image_metrics_and_images(outputs, batch)
    print("Metrics:", metrics)
    print("Images dict keys:", images.keys())
