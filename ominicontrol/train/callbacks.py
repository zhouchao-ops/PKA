import lightning as L
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
from transformers import pipeline
import cv2
import torch
import os

try:
    import wandb
except ImportError:
    wandb = None

from ..flux.condition import Condition
from ..flux.generate import generate


class TrainingCallback(L.Callback):
    def __init__(self, run_name, training_config: dict = {}):
        self.run_name, self.training_config = run_name, training_config

        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = training_config.get("save_path", "./output")

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = (
            wandb is not None and os.environ.get("WANDB_API_KEY") is not None
        )

        self.total_steps = 0
        self.last_saved_optimizer_step = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        self.total_steps += 1

        # Print training progress every n steps
        if self.use_wandb:
            report_dict = {
                "steps": batch_idx,
                "steps": self.total_steps,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
            }
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
            report_dict["loss"] = loss_value
            report_dict["t"] = pl_module.last_t
            wandb.log(report_dict)

        if self.total_steps % self.print_every_n_steps == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Batch: {batch_idx}, Loss: {pl_module.log_loss:.4f}, Gradient size: {gradient_size:.4f}, Max gradient size: {max_gradient_size:.4f}"
            )

        # Lightning global_step counts optimizer updates, unlike total_steps,
        # which counts micro-batches and therefore includes accumulation.
        optimizer_step = trainer.global_step
        if (
            optimizer_step > 0
            and optimizer_step % self.save_interval == 0
            and optimizer_step != self.last_saved_optimizer_step
        ):
            print(
                f"Epoch: {trainer.current_epoch}, Optimizer step: {optimizer_step} - Saving LoRA weights"
            )
            pl_module.save_lora(
                f"{self.save_path}/{self.run_name}/ckpt/{optimizer_step}"
            )
            self.last_saved_optimizer_step = optimizer_step

        # Generate and save a sample image at specified intervals
        # if self.total_steps % self.sample_interval == 0:
        #     print(
        #         f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Generating a sample"
        #     )
        #     self.generate_a_sample(
        #         trainer,
        #         pl_module,
        #         f"{self.save_path}/{self.run_name}/output/lora_{self.total_steps}",
        #         f"lora_{self.total_steps}",
        #         batch["condition_type"][
        #             0
        #         ],  # Use the condition type from the current batch
        #     )

    @torch.no_grad()
    def generate_a_sample(
        self,
        trainer,
        pl_module,
        save_path,
        file_name,
        condition_type="super_resolution",
    ):
        # TODO: change this two variables to parameters
        condition_size = trainer.training_config["dataset"]["condition_size"]
        target_size = trainer.training_config["dataset"]["target_size"]
        position_scale = trainer.training_config["dataset"].get("position_scale", 1.0)

        generator = torch.Generator(device=pl_module.device)
        generator.manual_seed(42)

        test_list = []

        # if condition_type == "subject":
        #     test_list.extend(
        #         [
        #             (
        #                 Image.open("assets/test_in.jpg"),
        #                 [0, -32],
        #                 "Resting on the picnic table at a lakeside campsite, it's caught in the golden glow of early morning, with mist rising from the water and tall pines casting long shadows behind the scene.",
        #             ),
        #             (
        #                 Image.open("assets/test_out.jpg"),
        #                 [0, -32],
        #                 "In a bright room. It is placed on a table.",
        #             ),
        #         ]
        #     )
        # elif condition_type == "canny" or condition_type == "fusion":
        #     condition_img = Image.open("assets/vase_hq.jpg").resize(
        #         (condition_size, condition_size)
        #     )
        #     condition_img = np.array(condition_img)
        #     condition_img = cv2.Canny(condition_img, 100, 200)
        #     condition_img = Image.fromarray(condition_img).convert("RGB")
        #     test_list.append(
        #         (
        #             condition_img,
        #             [0, 0],
        #             "A beautiful vase on a table.",
        #             {"position_scale": position_scale} if position_scale != 1.0 else {},
        #         )
        #     )
        # elif condition_type == "coloring":
        #     condition_img = (
        #         Image.open("assets/vase_hq.jpg")
        #         .resize((condition_size, condition_size))
        #         .convert("L")
        #         .convert("RGB")
        #     )
        #     test_list.append((condition_img, [0, 0], "A beautiful vase on a table."))
        # elif condition_type == "depth":
        #     if not hasattr(self, "deepth_pipe"):
        #         self.deepth_pipe = pipeline(
        #             task="depth-estimation",
        #             model="LiheYoung/depth-anything-small-hf",
        #             device="cpu",
        #         )
        #     condition_img = (
        #         Image.open("assets/vase_hq.jpg")
        #         .resize((condition_size, condition_size))
        #         .convert("RGB")
        #     )
        #     condition_img = self.deepth_pipe(condition_img)["depth"].convert("RGB")
        #     test_list.append(
        #         (
        #             condition_img,
        #             [0, 0],
        #             "A beautiful vase on a table.",
        #             {"position_scale": position_scale} if position_scale != 1.0 else {},
        #         )
        #     )
        # elif condition_type == "depth_pred":
        #     condition_img = (
        #         Image.open("assets/vase_hq.jpg")
        #         .resize((condition_size, condition_size))
        #         .convert("RGB")
        #     )
        #     test_list.append((condition_img, [0, 0], "A beautiful vase on a table."))
        # elif condition_type == "deblurring":
        #     blur_radius = 5
        #     image = Image.open("./assets/vase_hq.jpg")
        #     condition_img = (
        #         image.convert("RGB")
        #         .resize((condition_size, condition_size))
        #         .filter(ImageFilter.GaussianBlur(blur_radius))
        #         .convert("RGB")
        #     )
        #     test_list.append(
        #         (
        #             condition_img,
        #             [0, 0],
        #             "A beautiful vase on a table.",
        #             {"position_scale": position_scale} if position_scale != 1.0 else {},
        #         )
        #     )
        # elif condition_type == "fill":
        #     condition_img = (
        #         Image.open("./assets/vase_hq.jpg")
        #         .resize((condition_size, condition_size))
        #         .convert("RGB")
        #     )
        #     mask = Image.new("L", condition_img.size, 0)
        #     draw = ImageDraw.Draw(mask)
        #     a = condition_img.size[0] // 4
        #     b = a * 3
        #     draw.rectangle([a, a, b, b], fill=255)
        #     condition_img = Image.composite(
        #         condition_img, Image.new("RGB", condition_img.size, (0, 0, 0)), mask
        #     )
        #     test_list.append((condition_img, [0, 0], "A beautiful vase on a table."))
        # elif condition_type == "sr":
        #     condition_img = (
        #         Image.open("assets/vase_hq.jpg")
        #         .resize((condition_size, condition_size))
        #         .convert("RGB")
        #     )
        #     test_list.append((condition_img, [0, -16], "A beautiful vase on a table."))
        # elif condition_type == "cartoon":
        #     condition_img = (
        #         Image.open("assets/cartoon_boy.png")
        #         .resize((condition_size, condition_size))
        #         .convert("RGB")
        #     )
        #     test_list.append(
        #         (
        #             condition_img,
        #             [0, -16],
        #             "A cartoon character in a white background. He is looking right, and running.",
        #         )
        #     )
        # # else:
        # #     raise NotImplementedError
        # for i, (condition_img, position_delta, prompt, *others) in enumerate(test_list):
        #         condition = Condition(
        #             condition_type=condition_type,
        #             condition=condition_img.resize(
        #                 (condition_size, condition_size)
        #             ).convert("RGB"),
        #             position_delta=position_delta,
        #             **(others[0] if others else {}),
        #         )

        # gen
        if 1:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            images = [
                "coffee",
                "vase_hq",
                # "room_corner",
                # "clock",
                # "heart",
                ]
            prompts = [
                "In a bright room. A cup of a coffee with some beans on the side. They are placed on a dark wooden table.",
                "A beautiful vase on a table.",
                # "A sofa and a table are arranged side by side. There is a desk lamp on the table.",
                # "A yellow alarm clock, with yellow knitted fabric on the left and white silk fabric on the right.",
                # "art with heart challenge",
                ]
            for i,(image,prompt) in enumerate(zip(images,prompts)):
                image = Image.open(f"assets/{image}.jpg").convert("RGB")
                w, h, min_dim = image.size + (min(image.size),)
                image = image.crop(
                    ((w - min_dim) // 2, (h - min_dim) // 2, (w + min_dim) // 2, (h + min_dim) // 2)
                ).resize((512, 512))
                configured_spatial_types = trainer.training_config.get(
                    "condition_types", ["canny", "depth"]
                )
                include_subject = "subject" in trainer.training_config.get(
                    "condition_type", ""
                )
                compact_kwargs = (
                    {"position_scale": position_scale}
                    if position_scale != 1.0 else {}
                )
                condition_specs = []
                if include_subject:
                    condition_specs.append((
                        "subject",
                        image.resize((condition_size, condition_size)),
                        {"position_delta": [0, -condition_size // 16], **compact_kwargs},
                    ))
                condition_specs.extend(
                    (
                        condition_type,
                        image.resize((condition_size, condition_size)),
                        {"position_delta": [0, 0], **compact_kwargs},
                    )
                    for condition_type in configured_spatial_types
                )
                # The compact pair configs evaluate exactly the pair used for
                # training instead of generating every subset.
                all_combinations = [tuple(condition_specs)]
                
                # 为每个组合生成结果
                for combo in all_combinations:
                    conditions = [
                        Condition(ctype, condition=img, **kwargs)
                        for ctype, img, kwargs in combo
                    ]
                    result_img = generate(
                        pl_module.flux_pipe,
                        prompt=prompt,
                        conditions=conditions,
                        height=target_size,
                        width=target_size,
                        generator=generator,
                        model_config=pl_module.model_config,
                        default_lora=True,
                    ).images[0]
                    
                    # 创建拼接图像
                    concat_image = Image.new("RGB", (512*(len(conditions)+2), 512))
                    concat_image.paste(image, (0, 0))
                    for j, condition in enumerate(conditions):
                        concat_image.paste(condition.condition, (512*(j+1), 0))
                    concat_image.paste(result_img, (512*(len(conditions)+1), 0))
                    
                    # 保存结果，文件名包含使用的条件类型
                    combo_names = "_".join([ctype for ctype, _, _ in combo])
                    concat_image.save(
                        os.path.join(save_path, f"{file_name}_{combo_names}_{i}.jpg")
                    )


class ConditionStrategyCallback(L.Callback):
    def __init__(self,
        dataset_instance,
        training_phases,
    ):
        super().__init__()
        self.dataset_instance = dataset_instance
        self.training_phases = training_phases
        self.current_phase_idx = 0
        self.current_min_conditions = 0
        self.current_max_conditions = 1

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        # 在训练开始时，设置初始阶段的条件范围
        initial_phase = self.training_phases[0]
        self.current_min_conditions = initial_phase["min_conditions"]
        self.current_max_conditions = initial_phase["max_conditions"]
        self.dataset_instance.set_condition_range(self.current_min_conditions, self.current_max_conditions)
        print(f"Training started. Initial phase set: {initial_phase['name']} "
              f"with conditions [{self.current_min_conditions}-{self.current_max_conditions}]")

    def on_train_batch_start(self, trainer: L.Trainer, pl_module: L.LightningModule, batch, batch_idx):
        # 在每个训练批次开始前检查并更新条件范围
        global_step = trainer.global_step

        # 确保不会越界，并且当前步数达到了当前阶段的阈值
        if self.current_phase_idx < len(self.training_phases) - 1 and \
           global_step >= self.training_phases[self.current_phase_idx]["threshold_step"]:

            self.current_phase_idx += 1
            next_phase = self.training_phases[self.current_phase_idx]
            print(f"\n--- Global Step {global_step}: Transitioning to {next_phase['name']} ---")

            # 更新数据集的条件范围
            self.current_min_conditions = next_phase["min_conditions"]
            self.current_max_conditions = next_phase["max_conditions"]
            self.dataset_instance.set_condition_range(self.current_min_conditions, self.current_max_conditions)

    
