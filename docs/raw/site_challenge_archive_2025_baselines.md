# Baselines - BEHAVIOR

> 원본: https://behavior.stanford.edu/challenge/archive/2025/baselines.html
> 받은 시각: 2026-09-29 18:22 KST (2026-09-29 09:22 UTC)
> 이 파일은 `_scrape.py` 가 자동으로 받은 원문 보관본이다. HTML→Markdown 변환 중 서식은 달라질 수 있으나 문장은 원문 그대로다.

---
# Baselines

## Pi0

This tutorial provides a simplest version instruction to finetune Pi0 on the 2025 BEHAVIOR-1K Challenge dataset.

### Repo Clone

```
git clone https://github.com/StanfordVL/b1k-baselines.git --recurse-submodules
git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
```

 This finetuning instruction is adapted from the original [openpi repo](https://github.com/Physical-Intelligence/openpi).

### Installation

Openpi use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```
cd baselines/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

source .venv/bin/activate

# Install behavior for server deploy
cd $PATH_TO_BEHAVIOR_1K
uv pip install -e bddl
uv pip install -e OmniGibson[eval]
```

### Finetune OpenPi

We provide a Pi0 checkpoint for:

- turning_on_radio task [here](https://drive.google.com/file/d/1YU7evHBj7vfjmE-tNK-Rbie8ytholQTc/view?usp=sharing). This checkpoint has been trained for 50k steps.
- picking_up_trash task [here](https://drive.google.com/file/d/1G_ACu3uUP_9RmXDgqa7307aFt28G-vJN/view?usp=sharing). This checkpoint has been trained for 50k steps.

If you would like to run eval only feel free to skip to the last section.

Before we can run training, we need to compute the normalization statistics for the training data. Change line 98 of `compute_norm_stats.py` to be the task name you want (or None to include all tasks), then run the script below

```
uv run scripts/compute_norm_stats.py --config-name pi0_b1k
```

 This will create `norm_stats.json` under `assets/pi0_b1k/behavior-1k/2025-challenge-demos`

After this, change line 137 of `data_loader.py` to be the task name you want (or None to include all tasks), then run the following command to fintune OpenPi:

```
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_val.py pi0_b1k \
    --exp_name="openpi_$(date +%Y%m%d_%H%M%S)" \
    --overwrite \
    --batch_size=64 \
    --num_train_steps=50000 \
    --weight_loader.params_path=gs://openpi-assets/checkpoints/pi0_base/params
```

### Evaluation

After finetuning, you can run evaluation by following the steps below:

1. Deploy finetuned checkpoint:

  ```
  source .venv/bin/activate
  uv run scripts/serve_b1k.py --task_name=$TASK_NAME policy:checkpoint --policy.config=pi0_b1k --policy.dir=$PATH_TO_CKPT
  ```
   This opens a connection listening on 0.0.0.0:8000.
2. Run the evaluation on BEHAVIOR:

  Assume you have behavior env installed (check https://github.com/StanfordVL/BEHAVIOR-1K for more details), run the following command within the BEHAVIOR-1K directory:
  ```
  conda activate behavior
  python OmniGibson/omnigibson/learning/eval.py policy=websocket task.name=turning_on_radio log_path=$LOG_PATH
  ```

## OpenVLA-OFT

This tutorial provides a simplest version instruction to finetune OpenVLA-OFT on the 2025 BEHAVIOR-1K Challenge dataset.

### Repo Clone

```
git clone https://github.com/StanfordVL/b1k-baselines.git --recurse-submodules
git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
```

 You can also start with the original openvla-oft [repo](https://github.com/moojink/openvla-oft). This finetuning instruction is adapted from the [ALOHA finetuning task](https://github.com/moojink/openvla-oft/blob/main/ALOHA.md).

### Installation

```
conda create -n openvla-oft python=3.10 cudatoolkit-dev "setuptools<=79" -c conda-forge -y
conda activate openvla-oft

# Install behavior for server deploy
cd $PATH_TO_BEHAVIOR_1K
pip install -e bddl
pip install -e OmniGibson[eval]

cd $PATH_TO_OPENVLA_OFT
pip install -e .

pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation

# Install for RLDS dataset: tensorflow, tensorflow_datasets, tensorflow_hub, apache_beam
pip install tensorflow_hub apache_beam
```

### Data Conversion

Since Openvla-oft requires RLDS dataset, we need first convert BEHAVIOR dataset into RLDS format.

1. See instructions for converting to RLDS [here](https://github.com/evansh666/openvla-oft/blob/main/RLDS_builder/README.md).
2. A sample BEHAVIOR data to RLDS conversion script is available [here](https://github.com/evansh666/openvla-oft/tree/main/RLDS_builder/behavior_dataset/behavior_turn_on_radio), you can use the following code to get RLDS-formatted data:

```
cd RLDS_builder
tfds build --data_dir /path/to/save/rlds/dataset
```

1. If you want to customize your own dataset, revise the dataset builder (e.g., ['behavior_turn_on_radio_dataset_builder.py'](https://github.com/evansh666/openvla-oft/blob/main/RLDS_builder/behavior_dataset/behavior_turn_on_radio/behavior_turn_on_radio_dataset_builder.py)).

### Finetune OpenVLA-OFT+

There are a few files in OpenVLA-OFT+ needs change to adapt our new robot:

1. Register the dataset (e.g. behavior_turn_on_radio) with openvla-oft dataloader by adding an entry in the following files:

  - Add an entry in StateEncoding and ActionEncoding; and Add a data name mapping in `configs.py` ([here](https://github.com/evansh666/openvla-oft/tree/main/prismatic/vla/datasets/rlds/oxe/configs.py#L711))
  - Add data transform in `transforms.py` ([here](https://github.com/evansh666/openvla-oft/tree/main/prismatic/vla/datasets/rlds/oxe/transforms.py#L937))
  - Add data mixture proportion in `mixtures.py` ([here](https://github.com/evansh666/openvla-oft/tree/main/prismatic/vla/datasets/rlds/oxe/mixtures.py#L231)).
  - Set constants of BEHAVIOR, e.g., desired action chunk size ([here](https://behavior.stanford.edu/https:/raw.githubusercontent.com/StanfordVL/b1k-baselines/main/tutorials/https:/github.com/evansh666/openvla-oft/tree/main/prismatic/vla/constants.py))
  - Add normalize and absolute action mask in `materialize.py` ([here](https://github.com/evansh666/openvla-oft/tree/main/prismatic/vla/datasets/rlds/oxe/materialize.py)).
  - Add behavior in three camera views selection ([here](https://github.com/evansh666/openvla-oft/tree/main/prismatic/vla/datasets/datasets.py#L116))
2. Revise dataset and setting in [finetune.sh](https://github.com/evansh666/openvla-oft/blob/main/scripts/finetune.sh). For more detailed parameter selection, please refer [OpenVLA-Finetune Instruction](https://github.com/moojink/openvla-oft/blob/main/ALOHA.md).

```
torchrun --standalone --nnodes 1 --nproc-per-node X vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /PATH/TO/RLDS/DATASETS/DIR/ \
  --dataset_name /YOUR/DATASET/NAME \
  --run_root_dir /YOUR/CHECKPOINTS/AND/LOG/DIR/ \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 3 \
  --use_proprio True \
  --batch_size 4 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 50000 \
  --max_steps 100005 \
  --use_val_set True \
  --val_freq 10000 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "YOUR_WANDB_PROJECT" \
  --run_id_note parallel_dec--25_acts_chunk--continuous_acts--L1_regression--3rd_person_img--left_right_wrist_imgs--proprio_state--film
```

### Evaluation

After finetuning, you can deploy the 1. Deploy finetuned checkpoint:

```
python vla-scripts/deploy.py \
  --pretrained_checkpoint /PATH/TO/FINETUNED/MODEL/CHECKPOINT/DIR/ \
  --use_l1_regression True \
  --use_film True \
  --num_images_in_input 3 \
  --use_proprio True \
  --center_crop True \
  --unnorm_key /NAME/OF/DATASET

  # Or directly run after modifying deploy.sh
  ./scripts/deploy.sh
```

 This opens a connection listening on 0.0.0.0:8000.

1. Run the evaluation on BEHAVIOR:

Assume you have behavior env installed (check https://github.com/StanfordVL/BEHAVIOR-1K for more details), run the following command within the BEHAVIOR-1K directory:

```
conda activate behavior
python OmniGibson/omnigibson/learning/eval.py policy=websocket task.name=turning_on_radio log_path=$LOG_PATH
```

## Other IL Policies

This tutorial provides instruction to train various classic imitation learning policies on the 2025 BEHAVIOR-1K Challenge dataset.

### Repo Clone

```
git clone https://github.com/StanfordVL/b1k-baselines.git --recurse-submodules
git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
```

 This il_lib repo is inspired by Behavior Robot Suite [repo](https://github.com/behavior-robot-suite/brs-algo).

### Installation

IL_LIB is compatible with the BEHAIVOR-1K repo, so you can directly install

```
conda activate behavior
cd baselines/il_lib
pip install -e .
```

Alternatively, feel free to have it installed in a new conda env, although you need to have the complete behavior stack installed within this conda env to perform online evaluation during training.

### Model Training

We provide a trained WB-VIMA checkpoint [here](https://drive.google.com/file/d/1YTB0XCh32EHyq2svcsYN6HNidk7-8hkJ/view?usp=sharing) for the turning_on_radio task. If you would like to run eval only feel free to skip to the last section.

Note: in order to run WB-VIMA training, you will need to have point cloud generated locally, we provide a sample point cloud generated for `turning_on_radio` [here](https://behavior.stanford.edu/https:/raw.githubusercontent.com/StanfordVL/b1k-baselines/main/tutorials), for more information, checkout the `generate_custon_data.ipynb` within the b1k-baselines repo.

IL_LIB includes implementations for common behavior cloning baselines, including RGB(D) [Diffusion Poilcy](https://diffusion-policy.cs.columbia.edu/), [3D Diffusion Policy](https://3d-diffusion-policy.github.io/), [ACT](https://tonyzhaozh.github.io/aloha/), [BC-RNN](https://robomimic.github.io/), [WB-VIMA](https://behavior-robot-suite.github.io/). You can find all the config files under `il_lib/configs/arch`. We will use `WB-VIMA` as the example for the following tutorial.

Before running actual training, try perform a fast_dev_run to make sure everything is ok:

```
python train.py data_dir=$DATA_PATH robot=r1pro task=behavior task.name=turning_on_radio arch=wbvima trainer.fast_dev_run=true +eval=behavior headless=false
```

If the train & eval sanity check passes and you see OmniGibson starts online evaluation, you can safely exit the program, the run the following command to launch the actual training:

```
python train.py data_dir=$DATA_PATH robot=r1pro task=behavior task.name=turning_on_radio arch=wbvima
```

Overwrite any parameters in the CLI if needed.

### Evaluation

After finetuning, you can run evaluation by following the steps below:

1. Deploy finetuned checkpoint:

  ```
  python serve.py robot=r1pro task=behavior arch=wbvima ckpt_path=$CKPT_ATH
  ```
   This opens a connection listening on 0.0.0.0:8000.
2. Run the evaluation on BEHAVIOR

  Put `wbvima_wrapper.py` under `OmniGibson/omnigibson/learning/wrappers`, then run:
  ```
  conda activate behavior
  python OmniGibson/omnigibson/learning/eval.py policy=websocket task.name=turning_on_radio env_wrapper._target_=omnigibson.learning.wrappers.wbvima_wrapper.WBVIMAWrapper log_path=$LOG_PATH
  ```
