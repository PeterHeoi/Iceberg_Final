# Iceberg 最终方案

最终提交使用两套 XGBoost 的预测概率，SAR-HUB ResNet-18 分支占 40%，SAR-HUB ResNet-50 分支占 60%。对应 Kaggle Public log loss 为 **0.12138**，Private log loss 为 **0.09742**。

## 在当前电脑重跑

1. 将整个文件夹移到目标位置。在 PyCharm 中用 **File → Open** 打开移动后的文件夹，或在该文件夹的终端运行 `jupyter lab`。
2. 打开 `final_iceberg_multiscale_ben18w40_ben50w60_xgb150.ipynb`，选择已有的 `dl` 内核。内核工作目录应为 Notebook 所在文件夹。
3. 在首个代码单元选择 `MODE`，然后执行 **Run All Cells**。

| 模式 | 计算内容 | 生成 CSV 的目录 |
|---|---|---|
| `"cached"`（默认） | 从 CNN 缓存重新拟合两套 XGBoost、计算五折验证结果；从测试集两列分支概率重新计算最终融合，并校验历史提交的 SHA-256。 | `outputs/final/cached/` |
| `"full"` | 从训练图像运行两个 seed 的五折验证（80 次 CNN 训练），再以 seed 42 进行提交所需的 20 次 CNN 训练、测试集推理和融合。每次训练 24 个 epoch。 | `outputs/final/full/` |

两种模式均写出 `submission_multiscale_ben18w40_ben50w60_xgb150.csv`。完整重训优先使用 CUDA，缺少 CUDA 时使用 CPU。跨设备或软件版本重训的浮点结果可能略有差异。

## 换到新 Python 环境

原运行环境为 64 位 Python 3.12.15、PyTorch 2.14.1+cu132、torchvision 0.29.1+cu132。使用新环境时，可先安装与设备匹配的 PyTorch 和 torchvision，再在该文件夹运行以下命令，然后在 JupyterLab 中选择该环境的内核。

```console
python -m pip install -r requirements.txt
python -m ipykernel install --user --name iceberg-final --display-name "Python (Iceberg Final)"
jupyter lab
```

## 随附文件

| 目录 | 用途 |
|---|---|
| `data/` | 原始训练和测试 JSON、四方向增强训练 JSON、固定的五折划分，以及 Kaggle 提交模板。 |
| `weights/` | ImageNet ResNet-18 和两份 SAR-HUB BigEarthNet-S1 权重。Notebook 从文件夹内直接加载。 |
| `inputs/cached/` | 默认模式的三份验证缓存和一张测试集分支概率表。分支概率保留原始小数文本。 |
| `vendor/windows-x64/` | 原 Windows x64 环境使用的 XGBoost 3.4.1 和原包许可证。其他平台通过 requirements.txt 安装该版本。 |
| `outputs/final/` | `cached/` 和 `full/` 在交付时均为空。 |

> 注：`data/` 中的三个大型 JSON（`train.json`、`train_aug_hflip_vflip_rot180.json`、`test.json`）超出 GitHub 单文件 100MB 上限，未随仓库分发；克隆本仓库后请从 Kaggle 竞赛页下载并放入 `data/`，其余文件（`train_folds.csv`、`sample_submission.csv`、`inputs/cached/`、`weights/`）均已包含。
