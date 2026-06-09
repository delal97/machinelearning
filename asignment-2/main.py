import json
import torch

from PIL import Image
from torchvision import transforms
from torchvision.models import vgg16

from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
import matplotlib.pyplot as plt

with open("imagenet_class_index.json", "r") as f:
    classes = json.load(f)


# Model
model = vgg16(weights="DEFAULT")
model.eval()

cam_extractor = GradCAM(model, target_layer="features")


# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


img = Image.open("images/cat.jpg").convert("RGB")
input_tensor = transform(img).unsqueeze(0)


output = model(input_tensor)

pred = output.argmax().item()


# CAM
from torchvision.transforms.functional import to_pil_image

cams = cam_extractor(pred, output)

heatmap = to_pil_image(cams[0].cpu())

result = overlay_mask(img, heatmap, alpha=0.5)


# Show result
plt.imshow(result)
plt.axis("off")
plt.show()

result.save("results/cam_cat.png")


# Print prediction
print("\nPrediction:")
print(classes[str(pred)][1])

# Top 5
print("\nTop 5 predictions:\n")

values, indices = torch.topk(output, 5)

for idx, value in zip(indices[0], values[0]):
    idx = idx.item()
    print(classes[str(idx)][1], round(value.item(), 3))