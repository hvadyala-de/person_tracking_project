from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from torchreid import models


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REID_WEIGHTS = (
    PROJECT_ROOT
    / "models"
    / "reid"
    / "osnet_x0_25_msmt17.pth"
)


class ReIDExtractor:

    def __init__(self, device="cuda"):

        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )

        print(f"ReID device: {self.device}")

        if not REID_WEIGHTS.exists():
            raise FileNotFoundError(
                f"ReID weights not found: {REID_WEIGHTS}"
            )

        # ----------------------------------------------------
        # BUILD OSNET
        # ----------------------------------------------------
        #
        # This MSMT17 checkpoint has:
        #
        # classifier.weight = (4101, 512)
        #
        # So the model must be constructed with 4101 classes
        # before loading the checkpoint.
        #
        # pretrained=False is IMPORTANT:
        # we do NOT want ImageNet weights loaded first.
        # ----------------------------------------------------

        self.model = models.build_model(
            name="osnet_x0_25",
            num_classes=4101,
            pretrained=False,
        )

        # ----------------------------------------------------
        # LOAD MSMT17 PERSON-REID WEIGHTS
        # ----------------------------------------------------

        checkpoint = torch.load(
            REID_WEIGHTS,
            map_location="cpu",
            weights_only=True,
        )

        self.model.load_state_dict(
            checkpoint,
            strict=True,
        )

        print(
            f"Loaded MSMT17 ReID weights: {REID_WEIGHTS}"
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        # ----------------------------------------------------
        # PREPROCESSING
        # ----------------------------------------------------

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (256, 128)
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],
                ),
            ]
        )


    def extract(self, crop_bgr):

        if crop_bgr is None:
            return None

        if crop_bgr.size == 0:
            return None

        # OpenCV BGR -> RGB

        crop_rgb = cv2.cvtColor(
            crop_bgr,
            cv2.COLOR_BGR2RGB,
        )

        image = Image.fromarray(
            crop_rgb
        )

        tensor = self.transform(
            image
        )

        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(
            self.device
        )

        # ----------------------------------------------------
        # FEATURE EXTRACTION
        # ----------------------------------------------------

        with torch.no_grad():

            feature = self.model(
                tensor
            )

        feature = feature.squeeze(0)

        # ----------------------------------------------------
        # L2 NORMALIZATION
        # ----------------------------------------------------

        feature = torch.nn.functional.normalize(
            feature,
            p=2,
            dim=0,
        )

        return (
            feature
            .cpu()
            .numpy()
            .astype(np.float32)
        )
