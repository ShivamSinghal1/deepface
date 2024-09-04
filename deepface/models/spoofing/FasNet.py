from typing import Union, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from deepface.commons import folder_utils, file_utils


class CropImage:
    @staticmethod
    def _get_new_box(src_w, src_h, bbox, scale):
        x, y, box_w, box_h = bbox
        scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))

        # Ensure square aspect ratio
        new_size = max(box_w, box_h) * scale

        center_x, center_y = x + box_w / 2, y + box_h / 2

        left_top_x = center_x - new_size / 2
        left_top_y = center_y - new_size / 2
        right_bottom_x = center_x + new_size / 2
        right_bottom_y = center_y + new_size / 2

        # Adjust coordinates if they go outside the image
        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0
        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0
        if right_bottom_x > src_w - 1:
            left_top_x -= right_bottom_x - src_w + 1
            right_bottom_x = src_w - 1
        if right_bottom_y > src_h - 1:
            left_top_y -= right_bottom_y - src_h + 1
            right_bottom_y = src_h - 1

        return (
            int(left_top_x),
            int(left_top_y),
            int(right_bottom_x),
            int(right_bottom_y),
        )

    def crop(
        self,
        org_img: torch.Tensor,
        bbox,
        scale,
        out_w,
        out_h,
        crop=True,
        return_box=False,
    ):
        if not crop:
            dst_img = F.interpolate(
                org_img.unsqueeze(0),
                size=(out_h, out_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            return (dst_img, bbox) if return_box else dst_img
        else:
            _, src_h, src_w = org_img.shape
            left_top_x, left_top_y, right_bottom_x, right_bottom_y = self._get_new_box(
                src_w, src_h, bbox, scale
            )

            # Ensure coordinates are within image boundaries
            left_top_x = max(0, left_top_x)
            left_top_y = max(0, left_top_y)
            right_bottom_x = min(src_w, right_bottom_x)
            right_bottom_y = min(src_h, right_bottom_y)

            img = org_img[:, left_top_y:right_bottom_y, left_top_x:right_bottom_x]

            # If the crop results in an empty image, use the entire original image
            if img.numel() == 0:
                img = org_img
                left_top_x, left_top_y, right_bottom_x, right_bottom_y = (
                    0,
                    0,
                    src_w,
                    src_h,
                )

            dst_img = F.interpolate(
                img.unsqueeze(0),
                size=(out_h, out_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            if return_box:
                return dst_img, (left_top_x, left_top_y, right_bottom_x, right_bottom_y)
            else:
                return dst_img


class Fasnet:
    """
    Mini Face Anti Spoofing Net Library.
    """

    def __init__(self):
        try:
            import torch
        except Exception as err:
            raise ValueError(
                "You must install torch with `pip install pytorch` command to use face anti spoofing module"
            ) from err

        home = folder_utils.get_deepface_home()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.crop_image = CropImage()

        # Download pre-trained models if not installed yet
        file_utils.download_external_file(
            file_name="2.7_80x80_MiniFASNetV2.pth",
            exact_file_path=f"{home}/.deepface/weights/2.7_80x80_MiniFASNetV2.pth",
            url="https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth",
        )

        file_utils.download_external_file(
            file_name="4_0_0_80x80_MiniFASNetV1SE.pth",
            exact_file_path=f"{home}/.deepface/weights/4_0_0_80x80_MiniFASNetV1SE.pth",
            url="https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth",
        )

        from deepface.models.spoofing import FasNetBackbone

        self.first_model = FasNetBackbone.MiniFASNetV2(conv6_kernel=(5, 5)).to(
            self.device
        )
        self.second_model = FasNetBackbone.MiniFASNetV1SE(conv6_kernel=(5, 5)).to(
            self.device
        )

        self.load_model(
            self.first_model, f"{home}/.deepface/weights/2.7_80x80_MiniFASNetV2.pth"
        )
        self.load_model(
            self.second_model,
            f"{home}/.deepface/weights/4_0_0_80x80_MiniFASNetV1SE.pth",
        )

        self.first_model.eval()
        self.second_model.eval()

    def load_model(self, model, path):
        state_dict = torch.load(path, map_location=self.device)
        if next(iter(state_dict)).find("module.") >= 0:
            from collections import OrderedDict

            new_state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)

    def analyze(
        self, imgs: List[torch.Tensor], facial_areas: List[Union[list, tuple]]
    ) -> List[Tuple[bool, float]]:
        """
        Analyze a batch of images to determine if they are spoofed or not.

        Args:
            imgs (List[torch.Tensor]): List of pre-loaded images as tensors.
            facial_areas (List[list or tuple]): List of facial rectangle area coordinates with x, y, w, h respectively.

        Returns:
            List[Tuple[bool, float]]: A list of result tuples consisting of is_real and score for each image.
        """
        assert len(imgs) == len(
            facial_areas
        ), "The number of images must match the number of facial areas."

        first_imgs = []
        second_imgs = []

        for idx, (img, facial_area) in enumerate(zip(imgs, facial_areas)):
            x, y, w, h = facial_area

            # Crop and draw bounding boxes for first and second images
            first_img, first_box = self.crop_image.crop(
                img, (x, y, w, h), 2.7, 80, 80, return_box=True
            )
            second_img, second_box = self.crop_image.crop(
                img, (x, y, w, h), 4, 80, 80, return_box=True
            )

            first_imgs.append(first_img)
            second_imgs.append(second_img)

        first_imgs = torch.stack(first_imgs).to(self.device)
        second_imgs = torch.stack(second_imgs).to(self.device)

        with torch.no_grad():
            first_results = F.softmax(self.first_model(first_imgs), dim=1)
            second_results = F.softmax(self.second_model(second_imgs), dim=1)

        predictions = (first_results + second_results).cpu().numpy()

        results = []
        for prediction in predictions:
            label = int(np.argmax(prediction))
            is_real = bool(label == 1)
            score = float(prediction[label] / 2)
            results.append((is_real, score))

        return results
