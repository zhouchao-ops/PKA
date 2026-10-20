import torch
from typing import Optional, Union, List, Tuple
from diffusers.pipelines import FluxPipeline
from PIL import Image, ImageFilter
import numpy as np
import cv2

from .pipeline_tools import encode_images
import os 

condition_temporal = os.getenv("CONDITION_TEMPORAL","0")
if condition_temporal == '1':
    condition_dict = {
        "depth": 1,
        "canny": 2,
        "subject": 4,
        "coloring": 3,
        "deblurring": 5,
        "depth_pred": 6,
        "fill": 7,
        "sr": 8,
        "cartoon": 9,
    }
elif condition_temporal == '0':
    condition_dict = {
        "depth": 0,
        "canny": 0,
        "subject": 0,
        "sketch": 0,
        "normal": 0,
        "segmentation": 0,
        "albedo": 0,
        "irradiance": 0,
    }
elif condition_temporal == '100':
    condition_dict = {
        "depth": 101,
        "canny": 102,
        "subject": 103,
        "coloring": 104,
        "deblurring": 105,
        "depth_pred": 106,
        "fill": 107,
        "sr": 108,
        "cartoon": 109,
    }
else:
    raise ValueError("condition temporal illegal")

class Condition(object):
    def __init__(
        self,
        condition_type: str,
        raw_img: Union[Image.Image, torch.Tensor] = None,
        condition: Union[Image.Image, torch.Tensor] = None,
        mask=None,
        position_delta=None,
        position_scale=1.0,
    ) -> None:
        self.condition_type = condition_type
        assert raw_img is not None or condition is not None
        if raw_img is not None:
            self.condition = self.get_condition(condition_type, raw_img)
        else:
            self.condition = condition
        self.position_delta = position_delta
        self.position_scale = position_scale
        # TODO: Add mask support
        assert mask is None, "Mask not supported yet"

    def get_condition(
        self, condition_type: str, raw_img: Union[Image.Image, torch.Tensor]
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Returns the condition image.
        """
        spatials = ['sketch',
       'canny', 'depth', 'normal', 'segmentation', 'albedo', 'irradiance']
        # if condition_type == "depth":
        #     from transformers import pipeline

        #     depth_pipe = pipeline(
        #         task="depth-estimation",
        #         model="LiheYoung/depth-anything-small-hf",
        #         device="cuda",
        #     )
        #     source_image = raw_img.convert("RGB")
        #     condition_img = depth_pipe(source_image)["depth"].convert("RGB")
        #     return condition_img
        # elif condition_type == "canny":
        #     img = np.array(raw_img)
        #     edges = cv2.Canny(img, 100, 200)
        #     edges = Image.fromarray(edges).convert("RGB")
        #     return edges
        if condition_type == "subject" or condition_type in spatials:
            return raw_img.convert("RGB")
        elif condition_type == "coloring":
            return raw_img.convert("L").convert("RGB")
        elif condition_type == "deblurring":
            condition_image = (
                raw_img.convert("RGB")
                .filter(ImageFilter.GaussianBlur(10))
                .convert("RGB")
            )
            return condition_image
        elif condition_type == "fill":
            return raw_img.convert("RGB")
        elif condition_type == "cartoon":
            return raw_img.convert("RGB")
        return self.condition

    @property
    def type_id(self) -> int:
        """
        Returns the type id of the condition.
        """
        return condition_dict[self.condition_type]

    @classmethod
    def get_type_id(cls, condition_type: str) -> int:
        """
        Returns the type id of the condition.
        """
        return condition_dict[condition_type]

    def encode(
        self, pipe: FluxPipeline, empty: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Encodes the condition into tokens, ids and type_id.
        """
        if self.condition_type in [
            "depth",
            "canny",
            "subject",
            "coloring",
            "deblurring",
            "depth_pred",
            "fill",
            "sr",
            "cartoon",
        ]+['sketch',
       'canny', 'depth', 'normal', 'segmentation', 'albedo', 'irradiance']:
            if empty:
                # make the condition black
                e_condition = Image.new("RGB", self.condition.size, (0, 0, 0))
                e_condition = e_condition.convert("RGB")
                tokens, ids = encode_images(pipe, e_condition)
            else:
                tokens, ids = encode_images(pipe, self.condition)
        
        else:
            raise NotImplementedError(
                f"Condition type {self.condition_type} not implemented"
            )
        if self.position_delta is None and self.condition_type == "subject":
            self.position_delta = [0, -self.condition.size[0] // 16]
        if self.position_delta is not None:
            ids[:, 1] += self.position_delta[0]
            ids[:, 2] += self.position_delta[1]
        if self.position_scale != 1.0:
            scale_bias = (self.position_scale - 1.0) / 2
            ids[:, 1] *= self.position_scale
            ids[:, 2] *= self.position_scale
            ids[:, 1] += scale_bias
            ids[:, 2] += scale_bias
        type_id = torch.ones_like(ids[:, :1]) * self.type_id
        return tokens, ids, type_id
