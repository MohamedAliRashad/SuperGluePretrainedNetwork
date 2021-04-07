from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw, ImageOps
from torch import nn
from torchvision.transforms import ToTensor


def display_keypoints(img, keypoints):
    """ draw keypoints on img, img must be read in pillow """
    draw = ImageDraw.Draw(img)
    for key in keypoints:
        x, y = key
        r = 5
        leftUpPoint = (x - r, y - r)
        rightDownPoint = (x + r, y + r)
        twoPointList = [leftUpPoint, rightDownPoint]
        draw.ellipse(twoPointList, fill=(255, 0, 0))

    img.show()


def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert nms_radius >= 0

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],).to(
        keypoints
    )[None]
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    args = {"align_corners": True} if int(torch.__version__[2]) > 2 else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", **args
    )
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1
    )
    return descriptors


class SuperPoint(nn.Module):
    """
    Modified Version of SuperPoint that computes descriptors only
    """

    default_config = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": -1,
        "remove_borders": 4,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.config["descriptor_dim"], kernel_size=1, stride=1, padding=0
        )

        path = Path(__file__).parent / "weights/superpoint_v1.pth"
        self.load_state_dict(torch.load(str(path)))

        mk = self.config["max_keypoints"]
        if mk == 0 or mk < -1:
            raise ValueError('"max_keypoints" must be positive or "-1"')

        print("Loaded SuperPoint model")

    def forward(self, data, heat_map=False):
        """ Compute descriptors for image """
        # Shared Encoder
        x = self.relu(self.conv1a(data["image"]))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        if heat_map:
            return torch.nn.functional.interpolate(descriptors, scale_factor=8)

        # Extract descriptors
        descriptors = [
            sample_descriptors(k[None], d[None], 8)[0]
            for k, d in zip(data["keypoints"], descriptors)
        ]

        return descriptors


if __name__ == "__main__":
    sp = SuperPoint(SuperPoint.default_config)
    img_path = Path(__file__).parent / "sample.jpg"  # feel free to change image path
    img = Image.open(img_path)
    img.thumbnail(
        (800, 600), Image.ANTIALIAS
    )  # shrinking the image to fit in memory (can be commented if required)

    sample = ToTensor()(ImageOps.grayscale(img))[
        None, :
    ]  # to make the input image one channel and 4D
    keypoints = torch.randint(0, 600, (1, 80, 2))
    descriptors = sp({"image": sample, "keypoints": keypoints})[0]

    # keypoints need to be in shape of (N, xy)
    # display_keypoints(img, keypoints[0])

    ######### Uncomment if you want to visualize the descriptors' feature map over original image #########
    heat_map = (
        sp({"image": sample, "keypoints": keypoints}, heat_map=True)[0]
        .mean(0)
        .detach()
        .numpy()
    )
    f, axarr = plt.subplots(1, 2, figsize=(10, 4))
    axarr[0].imshow(img)
    axarr[0].set_title("Original")
    axarr[1].imshow(heat_map)
    axarr[1].set_title("Descriptors Map")

    # plt.imshow(heat_map)
    plt.show()
