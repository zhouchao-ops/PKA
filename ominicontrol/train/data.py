from PIL import Image, ImageFilter, ImageDraw
import cv2
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import random


class Subject200KDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        condition_size: int = 512,
        target_size: int = 512,
        image_size: int = 512,
        padding: int = 0,
        condition_type: str = "subject",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        return_pil_image: bool = False,
    ):
        self.base_dataset = base_dataset
        self.condition_size = condition_size
        self.target_size = target_size
        self.image_size = image_size
        self.padding = padding
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.base_dataset) * 2

    def __getitem__(self, idx):
        # If target is 0, left image is target, right image is condition
        target = idx % 2
        item = self.base_dataset[idx // 2]

        # Crop the image to target and condition
        image = item["image"]
        left_img = image.crop(
            (
                self.padding,
                self.padding,
                self.image_size + self.padding,
                self.image_size + self.padding,
            )
        )
        right_img = image.crop(
            (
                self.image_size + self.padding * 2,
                self.padding,
                self.image_size * 2 + self.padding * 2,
                self.image_size + self.padding,
            )
        )

        # Get the target and condition image
        target_image, condition_img = (
            (left_img, right_img) if target == 0 else (right_img, left_img)
        )

        # Resize the image
        condition_img = condition_img.resize(
            (self.condition_size, self.condition_size)
        ).convert("RGB")
        target_image = target_image.resize(
            (self.target_size, self.target_size)
        ).convert("RGB")

        # Get the description
        description = item["description"][
            "description_0" if target == 0 else "description_1"
        ]

        # Randomly drop text or image
        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob
        if drop_text:
            description = ""
        if drop_image:
            condition_img = Image.new(
                "RGB", (self.condition_size, self.condition_size), (0, 0, 0)
            )

        return {
            "image": self.to_tensor(target_image),
            "condition": self.to_tensor(condition_img),
            "condition_type": self.condition_type,
            "description": description,
            # 16 is the downscale factor of the image
            "position_delta": np.array([0, -self.condition_size // 16]),
            **({"pil_image": image} if self.return_pil_image else {}),
        }

AVAILABLE_SPATIAL_CONDITION_TYPES = [
    "sketch",
    "canny",
    "depth",
    "normal",
    "segmentation",
    "albedo",
    "irradiance",
]
import random
class Allcond200KDataset(Dataset):
    def __init__(
        self,
        base_dataset, # 你的基础数据集，例如经过过滤的 Yuanshi/Subjects200K dataset["train"]
        condition_size: int = 512, # 单个条件向量的维度 (如果你条件是文本，这是文本编码的维度)
        target_size: int = 512,  # 目标图像的尺寸 (长宽)
        image_size: int = 512,   # 图像处理后的尺寸
        padding: int = 0,        # 图像padding值 (这里简化为0，可能不使用)
        drop_text_prob: float = 0.1, # 文本条件丢弃概率 (这里可能不直接用到，取决于条件类型)
        drop_image_prob: float = 0.1, # 图像条件丢弃概率 (这里可能不直接用到，取决于条件类型)
        total_condition_slots: int = 7, # 条件槽位总数，默认7个
        spatial_condition_types = None,
        return_pil_image: bool = False, # 是否返回PIL图像而不是Tensor
        condition_type = "subject and spatial",
        position_scale: float = 1.0,
    ):
        self.base_dataset = base_dataset
        self.condition_size = condition_size
        self.target_size = target_size
        self.image_size = image_size
        self.padding = padding
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        if spatial_condition_types is None:
            spatial_condition_types = AVAILABLE_SPATIAL_CONDITION_TYPES[:total_condition_slots]
        else:
            spatial_condition_types = list(spatial_condition_types)
            if total_condition_slots != len(spatial_condition_types):
                raise ValueError(
                    "dataset.total_condition_slots must match the number of entries in "
                    f"train.condition_types ({total_condition_slots} != {len(spatial_condition_types)})"
                )
        if not spatial_condition_types:
            raise ValueError("At least one spatial condition type must be configured")
        unknown_types = set(spatial_condition_types) - set(AVAILABLE_SPATIAL_CONDITION_TYPES)
        if unknown_types:
            raise ValueError(f"Unsupported spatial condition types: {sorted(unknown_types)}")
        if len(spatial_condition_types) != len(set(spatial_condition_types)):
            raise ValueError("Duplicate entries in train.condition_types are not allowed")
        self.spatial_condition_types = spatial_condition_types
        self.return_pil_image = return_pil_image
        self.to_tensor = T.ToTensor()
        # 动态条件数量的存储位置，将在训练循环中被设置
        self._current_min_conditions = 0
        self._current_max_conditions = 1
        self.condition_type = condition_type
        self.position_scale = position_scale
        self.include_subject = "subject" in condition_type or condition_type == "fusion"
        self.include_spatial = "spatial" in condition_type or condition_type == "fusion"
        if not (self.include_subject or self.include_spatial):
            raise ValueError(
                "condition_type must contain 'subject' or 'spatial', or be 'fusion'"
            )

    def __len__(self):
        return len(self.base_dataset) * 2
    
    def image_split(self, image):
        left_img = image.crop(
            (
                self.padding,
                self.padding,
                self.image_size + self.padding,
                self.image_size + self.padding,
            )
        )
        right_img = image.crop(
            (
                self.image_size + self.padding * 2,
                self.padding,
                self.image_size * 2 + self.padding * 2,
                self.image_size + self.padding,
            )
        )
        return left_img, right_img

    def set_condition_range(self, min_num: int, max_num: int):
        # 设置condition的最大最小值，不包含subject
        if min_num < 0 or max_num < min_num:
            raise ValueError(f"Invalid condition range: [{min_num}, {max_num}]")
        if self.include_spatial and max_num > len(self.spatial_condition_types):
            raise ValueError(
                f"Condition range requests {max_num} spatial conditions, but only "
                f"{len(self.spatial_condition_types)} are configured: "
                f"{self.spatial_condition_types}"
            )
        self._current_min_conditions = min_num
        self._current_max_conditions = max_num
        print(f"Dataset condition range updated to [{self._current_min_conditions}, {self._current_max_conditions}]")

    def __getitem__(self, idx):
        target = idx % 2
        item = self.base_dataset[idx // 2]

        # Crop the image to target and condition
        image = item["image"]

        conditions = [item[condition_type] for condition_type in self.spatial_condition_types]
        if target == 0:
            target_image, subject_img = self.image_split(image)
            condition_imgs = [subject_img]
            condition_imgs += [self.image_split(condition)[0] for condition in conditions]
        else:
            subject_img, target_image = self.image_split(image)
            condition_imgs = [subject_img]
            condition_imgs += [self.image_split(condition)[1] for condition in conditions]
        
        # Resize the image
        target_image = target_image.resize(
            (self.target_size, self.target_size)
        ).convert("RGB")
        condition_imgs = [condition_img.resize(
            (self.condition_size, self.condition_size)
        ).convert("RGB") for condition_img in condition_imgs]
        
        # Get the description
        description = item["description"][
            "description_0" if target == 0 else "description_1"
        ]

        # to do drop condition 
        condition_num = random.randint(self._current_min_conditions, self._current_max_conditions)
        selected_conditions_indices = random.sample(range(1,len(condition_imgs)),condition_num)
        selected_conditions = [condition_imgs[0]] if self.include_subject else []
        selected_condition_types = ["subject"] if self.include_subject else []
        if self.include_spatial:
            selected_conditions += [condition_imgs[i] for i in selected_conditions_indices]
            selected_condition_types += [self.spatial_condition_types[i-1] for i in selected_conditions_indices]
        drop_text = random.random() < self.drop_text_prob
        if drop_text:
            description = ""
        for i in range(len(selected_conditions)):
            drop_image = random.random() < self.drop_image_prob
            if drop_image:
                condition_img = Image.new(
                    "RGB", (self.condition_size, self.condition_size), (0, 0, 0)
                )
                selected_conditions[i] = condition_img
        
        conditions = [self.to_tensor(item) for item in selected_conditions]
        position_delta = [np.array([0, -self.condition_size // 16])] if self.include_subject else []
        if self.include_spatial:
            position_delta += [np.array([0, 0])] * condition_num

            
        has_subject = self.include_subject
        return {
            "image": self.to_tensor(target_image),
            "condition": conditions,
            "condition_type": selected_condition_types,
            "description": description,
            # 16 is the downscale factor of the image
            "position_delta": position_delta,
            "has_subject":str(has_subject),
            **({"pil_image": image} if self.return_pil_image else {}),
            **({"position_scale": self.position_scale} if self.position_scale != 1.0 else {}),
        }


class ImageConditionDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        condition_size: int = 512,
        target_size: int = 512,
        condition_type: str = "canny",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        return_pil_image: bool = False,
        position_scale=1.0,
    ):
        self.base_dataset = base_dataset
        self.condition_size = condition_size
        self.target_size = target_size
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image
        self.position_scale = position_scale

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.base_dataset)

    def set_condition_range(self, min_num: int, max_num: int):
        """Validate the shared curriculum API for this fixed single-condition dataset."""
        if (min_num, max_num) != (1, 1):
            raise ValueError(
                "ImageConditionDataset always supplies exactly one condition; "
                f"received range [{min_num}, {max_num}]"
            )

    @property
    def depth_pipe(self):
        if not hasattr(self, "_depth_pipe"):
            from transformers import pipeline

            self._depth_pipe = pipeline(
                task="depth-estimation",
                model="LiheYoung/depth-anything-small-hf",
                device="cpu",
            )
        return self._depth_pipe

    def _get_canny_edge(self, img):
        resize_ratio = self.condition_size / max(img.size)
        img = img.resize(
            (int(img.size[0] * resize_ratio), int(img.size[1] * resize_ratio))
        )
        img_np = np.array(img)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img_gray, 100, 200)
        return Image.fromarray(edges).convert("RGB")

    def __getitem__(self, idx):
        image = self.base_dataset[idx]["jpg"]
        image = image.resize((self.target_size, self.target_size)).convert("RGB")
        description = self.base_dataset[idx]["json"]["prompt"]

        enable_scale = random.random() < 1
        if not enable_scale:
            condition_size = int(self.condition_size * self.position_scale)
            position_scale = 1.0
        else:
            condition_size = self.condition_size
            position_scale = self.position_scale

        # Get the condition image
        position_delta = np.array([0, 0])
        if self.condition_type == "canny":
            condition_img = self._get_canny_edge(image)
        elif self.condition_type == "coloring":
            condition_img = (
                image.resize((condition_size, condition_size))
                .convert("L")
                .convert("RGB")
            )
        elif self.condition_type == "deblurring":
            blur_radius = random.randint(1, 10)
            condition_img = (
                image.convert("RGB")
                .filter(ImageFilter.GaussianBlur(blur_radius))
                .resize((condition_size, condition_size))
                .convert("RGB")
            )
        elif self.condition_type == "depth":
            condition_img = self.depth_pipe(image)["depth"].convert("RGB")
            condition_img = condition_img.resize((condition_size, condition_size))
        elif self.condition_type == "depth_pred":
            condition_img = image
            image = self.depth_pipe(condition_img)["depth"].convert("RGB")
            description = f"[depth] {description}"
        elif self.condition_type == "fill":
            condition_img = image.resize((condition_size, condition_size)).convert(
                "RGB"
            )
            w, h = image.size
            x1, x2 = sorted([random.randint(0, w), random.randint(0, w)])
            y1, y2 = sorted([random.randint(0, h), random.randint(0, h)])
            mask = Image.new("L", image.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([x1, y1, x2, y2], fill=255)
            if random.random() > 0.5:
                mask = Image.eval(mask, lambda a: 255 - a)
            condition_img = Image.composite(
                image, Image.new("RGB", image.size, (0, 0, 0)), mask
            )
        elif self.condition_type == "sr":
            condition_img = image.resize((condition_size, condition_size)).convert(
                "RGB"
            )
            position_delta = np.array([0, -condition_size // 16])

        elif self.condition_type == "fusion":
            condition_img_canny = self._get_canny_edge(image)
            condition_img_coloring = (
                            image.resize((condition_size, condition_size))
                            .convert("L")
                            .convert("RGB")
                        )
            blur_radius = random.randint(1, 10)
            condition_img_blur = (
                            image.convert("RGB")
                            .filter(ImageFilter.GaussianBlur(blur_radius))
                            .resize((condition_size, condition_size))
                            .convert("RGB")
                        )
            condition_img = self.depth_pipe(image)["depth"].convert("RGB")
            condition_img_depth = condition_img.resize((condition_size, condition_size))
            condition_imgs = {
                "canny":condition_img_canny,
                # "coloring":condition_img_coloring,
                # "deblurring":condition_img_blur,
                "depth":condition_img_depth}
        else:
            raise ValueError(f"Condition type {self.condition_type} not implemented")

        # Randomly drop text or image
        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob
        if drop_text:
            description = ""
        if drop_image:
            condition_img = Image.new(
                "RGB", (condition_size, condition_size), (0, 0, 0)
            )

        if self.condition_type == "fusion":
            p = self.drop_image_prob
            condition_imgs_droped = {
                k: v for k, v in condition_imgs.items() 
                if random.random() > p
            }
            if not condition_imgs_droped:
            #     condition_imgs_droped = {
            #         k:  Image.new(
            #     "RGB", (condition_size, condition_size), (0, 0, 0)
            # ) for k in condition_imgs.keys() 
            #     }
                key = random.choice(list(condition_imgs.keys()))
                condition_imgs_droped = {key: condition_imgs[key]}
                
            conditions = [self.to_tensor(item) for item in condition_imgs_droped.values()]
            return {
                "image": self.to_tensor(image),
                "condition": conditions,
                "condition_type": [key for key in condition_imgs_droped.keys()],
                "description": description,
                "position_delta": position_delta,
                **({"pil_image": [image, *condition_imgs_droped.values()]} if self.return_pil_image else {}),
                **({"position_scale": position_scale} if position_scale != 1.0 else {}),
            }
        else:
            return {
                "image": self.to_tensor(image),
                "condition": [self.to_tensor(condition_img)],
                "condition_type": [self.condition_type],
                "description": description,
                "position_delta": [position_delta],
                **({"pil_image": [image, condition_img]} if self.return_pil_image else {}),
                **({"position_scale": position_scale} if position_scale != 1.0 else {}),
            }


class CartoonDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        condition_size: int = 1024,
        target_size: int = 1024,
        image_size: int = 1024,
        padding: int = 0,
        condition_type: str = "cartoon",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        return_pil_image: bool = False,
    ):
        self.base_dataset = base_dataset
        self.condition_size = condition_size
        self.target_size = target_size
        self.image_size = image_size
        self.padding = padding
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        condition_img = data["condition"]
        target_image = data["target"]

        # Tag
        tag = data["tags"][0]

        target_description = data["target_description"]

        description = {
            "lion": "lion like animal",
            "bear": "bear like animal",
            "gorilla": "gorilla like animal",
            "dog": "dog like animal",
            "elephant": "elephant like animal",
            "eagle": "eagle like bird",
            "tiger": "tiger like animal",
            "owl": "owl like bird",
            "woman": "woman",
            "parrot": "parrot like bird",
            "mouse": "mouse like animal",
            "man": "man",
            "pigeon": "pigeon like bird",
            "girl": "girl",
            "panda": "panda like animal",
            "crocodile": "crocodile like animal",
            "rabbit": "rabbit like animal",
            "boy": "boy",
            "monkey": "monkey like animal",
            "cat": "cat like animal",
        }

        # Resize the image
        condition_img = condition_img.resize(
            (self.condition_size, self.condition_size)
        ).convert("RGB")
        target_image = target_image.resize(
            (self.target_size, self.target_size)
        ).convert("RGB")

        # Process datum to create description
        description = data.get(
            "description",
            f"Photo of a {description[tag]} cartoon character in a white background. Character is facing {target_description['facing_direction']}. Character pose is {target_description['pose']}.",
        )

        # Randomly drop text or image
        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob
        if drop_text:
            description = ""
        if drop_image:
            condition_img = Image.new(
                "RGB", (self.condition_size, self.condition_size), (0, 0, 0)
            )

        return {
            "image": self.to_tensor(target_image),
            "condition": self.to_tensor(condition_img),
            "condition_type": self.condition_type,
            "description": description,
            # 16 is the downscale factor of the image
            "position_delta": np.array([0, -16]),
        }
