import io
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_and_preprocess_image(img_path):
    image = Image.open(img_path).convert("RGB")
    orig_size = image.size
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = transform(image)
    return image, orig_size


def predict_saliency_map(img, condition, model, device):
    img = img.unsqueeze(0).to(device)
    one_hot_condition = torch.zeros((1, 4), device=device)
    one_hot_condition[0, condition] = 1
    model.eval()
    with torch.no_grad():
        pred_saliency = model(img, one_hot_condition)

    pred_saliency = pred_saliency.squeeze().cpu().numpy()
    return pred_saliency


def overlay_heatmap_on_image(original_img_path, heatmap_img_path, output_img_path):
    # Read the original image
    orig_image = cv2.imread(original_img_path)
    orig_size = orig_image.shape[:2]  # Height, Width

    # Read the heatmap image
    overlay_heatmap = cv2.imread(heatmap_img_path, cv2.IMREAD_GRAYSCALE)

    # Resize the heatmap to match the original image size
    overlay_heatmap = cv2.resize(overlay_heatmap, (orig_size[1], orig_size[0]))

    # Apply color map to the heatmap
    overlay_heatmap = cv2.applyColorMap(overlay_heatmap, cv2.COLORMAP_JET)

    # Overlay the heatmap on the original image
    overlay_image = cv2.addWeighted(orig_image, 1, overlay_heatmap, 0.8, 0)

    # Save the result
    cv2.imwrite(output_img_path, overlay_image)


import io
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch


def write_heatmap_to_image(heatmap, orig_size, output_filename):
    # If heatmap is a torch tensor, convert to numpy.
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()
    # If heatmap has an extra channel dimension, remove it.
    if heatmap.ndim == 3 and heatmap.shape[0] == 1:
        heatmap = heatmap.squeeze(0)

    # Create a new figure, display the heatmap using the "hot" colormap and remove axes.
    plt.figure()
    plt.imshow(heatmap, cmap="hot")
    plt.axis("off")

    # Save the figure to an in-memory buffer in PNG format.
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    plt.close()

    # Load the image from the buffer using PIL.
    img = Image.open(buf)
    # Convert the image from RGBA (Matplotlib default) to BGR (OpenCV convention).
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)
    # Resize the image to the original size.
    img_resized = cv2.resize(img_cv, orig_size, interpolation=cv2.INTER_AREA)
    # Save the final image to disk.
    cv2.imwrite(output_filename, img_resized)
