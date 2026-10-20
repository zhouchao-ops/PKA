from torch.utils.data import DataLoader
import torch
import lightning as L
import yaml
import os
import time

from datasets import load_dataset

from .data import ImageConditionDataset, Subject200KDataset, CartoonDataset, Allcond200KDataset
from .model import OminiModel
from .callbacks import TrainingCallback, ConditionStrategyCallback 

def set_env(env):
    for k, v in env.items():
        os.environ[k] = str(v)

def get_rank():
    try:
        rank = int(os.environ.get("LOCAL_RANK"))
    except:
        rank = 0
    return rank


def get_config():
    config_path = os.environ.get("XFL_CONFIG")
    assert config_path is not None, "Please set the XFL_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # Keep release configs portable while allowing local training locations.
    flux_path = os.environ.get("OMINI_FLUX_PATH")
    if flux_path:
        config["flux_path"] = flux_path
    dataset_path = os.environ.get("OMINI_DATASET_PATH")
    if dataset_path:
        config["train"]["dataset"]["path"] = dataset_path
    save_path = os.environ.get("OMINI_SAVE_PATH")
    if save_path:
        config["train"]["save_path"] = save_path
    max_steps = os.environ.get("OMINI_MAX_STEPS")
    if max_steps:
        config["training_strategy"][-1]["threshold_step"] = int(max_steps)
    return config


def init_wandb(wandb_config, run_name):
    import wandb

    try:
        assert os.environ.get("WANDB_API_KEY") is not None
        wandb.init(
            project=wandb_config["project"],
            name=run_name,
            config={},
        )
    except Exception as e:
        print("Failed to initialize WanDB:", e)


def main():
    # Initialize
    is_main_process, rank = get_rank() == 0, get_rank()
    torch.cuda.set_device(rank)
    config = get_config()
    training_config = config["train"]
    run_name = time.strftime("%Y%m%d-%H%M%S")
    # set environment variables
    set_env(config["env"])
    # Initialize WanDB
    wandb_config = training_config.get("wandb", None)
    if wandb_config is not None and is_main_process:
        init_wandb(wandb_config, run_name)

    print("Rank:", rank)
    if is_main_process:
        print("Config:", config)


    # Initialize model
    trainable_model = OminiModel(
        flux_pipe_id=config["flux_path"],
        lora_config=training_config["lora_config"],
        lora_path=config.get("lora_path",None),
        device=f"cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
    )



    # Initialize dataset and dataloader
    if training_config["dataset"]["type"] == "img":
        dataset = load_dataset(
            "webdataset",
            data_files={"train": training_config["dataset"]["urls"]},
            split="train",
            cache_dir="cache/t2i2m",
            num_proc=32,
        )
        dataset = ImageConditionDataset(
            dataset,
            condition_size=training_config["dataset"]["condition_size"],
            target_size=training_config["dataset"]["target_size"],
            condition_type=training_config["condition_type"],
            drop_text_prob=training_config["dataset"]["drop_text_prob"],
            drop_image_prob=training_config["dataset"]["drop_image_prob"],
            position_scale=training_config["dataset"].get("position_scale", 1.0),
        )
    elif training_config["dataset"]["type"] in {"all", "all_high"}:
        dataset_path = training_config["dataset"].get("path")
        if not dataset_path:
            raise ValueError("train.dataset.path is required for all/all_high datasets")
        dataset = load_dataset(dataset_path, split="train")
        # Define filter function
        def filter_func(item):
            if not item.get("quality_assessment"):
                return False
            return all(
                item["quality_assessment"].get(key, 0) >= 5
                for key in ["compositeStructure", "objectConsistency", "imageQuality"]
            )
        os.makedirs("./cache/dataset",exist_ok=True)
        data_valid = dataset.filter(
            filter_func,
            num_proc=32,
            cache_file_name=training_config["dataset"].get(
                "cache_file", "./cache/dataset/all_conds.arrow"
            ),
        )
        dataset = Allcond200KDataset(
            data_valid,
            condition_size=training_config["dataset"]["condition_size"],
            target_size=training_config["dataset"]["target_size"],
            image_size=training_config["dataset"]["image_size"],
            padding=training_config["dataset"]["padding"],
            condition_type=training_config["condition_type"],
            drop_text_prob=training_config["dataset"]["drop_text_prob"],
            drop_image_prob=training_config["dataset"]["drop_image_prob"],
            total_condition_slots = training_config["dataset"]["total_condition_slots"], 
            spatial_condition_types=training_config.get("condition_types"),
            position_scale=training_config["dataset"].get("position_scale", 1.0),
        )
    else:
        raise NotImplementedError

    print("Dataset length:", len(dataset))

    # training strategy be like
    """
    training_strategy:
      - name: "Phase 1: Single Condition"
        min_conditions: 1
        max_conditions: 1
        threshold_step: 30000 # 训练到3万步

      - name: "Phase 2: 1 to 3 Conditions"
        min_conditions: 1
        max_conditions: 3
        threshold_step: 60000 # 训练到6万步 (从0开始累积)

      - name: "Phase 3: 3 to 7 Conditions"
        min_conditions: 3
        max_conditions: 7
        threshold_step: 90000 # 训练到9万步 (从0开始累积)
    """ 

        # 获取训练策略配置
    training_strategy = config.get("training_strategy", [])
    if not training_strategy:
        raise ValueError("training_strategy must be defined in your config.yaml")

    # DataLoader workers may prefetch samples before the strategy callback runs.
    # Initialize phase 1 here so spatial-only training never yields zero conditions.
    first_phase = training_strategy[0]
    dataset.set_condition_range(
        first_phase["min_conditions"], first_phase["max_conditions"]
    )


    train_loader = DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config["dataloader_workers"],
        pin_memory=True, # 提高数据传输效率
    )

    training_callbacks = []
    # 传入数据集实例和训练策略
    training_callbacks.append(ConditionStrategyCallback(
        dataset_instance=dataset, # 将数据集实例传递给 Callback
        training_phases=training_strategy
    ))
    if is_main_process:
        training_callbacks.append(TrainingCallback(run_name, training_config=training_config))

    max_steps = training_strategy[-1]["threshold_step"] if training_strategy else -1

    # Initialize trainer
    trainer = L.Trainer(
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        callbacks=training_callbacks,
        enable_checkpointing=False,
        enable_progress_bar=True,
        logger=False,
        max_steps=max_steps,
        max_epochs=training_config.get("max_epochs", -1),
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
        # 添加分布式训练相关参数 (如果你的环境是DDP)
        accelerator="gpu",
        devices="auto", # 自动检测可用 GPU
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto", # 如果多GPU，使用DDP
    )

    setattr(trainer, "training_config", training_config)

    # Save config
    save_path = training_config.get("save_path", "./output")
    if is_main_process:
        os.makedirs(f"{save_path}/{run_name}")
        with open(f"{save_path}/{run_name}/config.yaml", "w") as f:
            yaml.dump(config, f)

    # Start training
    trainer.fit(trainable_model, train_loader)


if __name__ == "__main__":
    main()
