# Uni-NaVid

**面向统一具身导航任务的视频视觉-语言-动作模型。** 本项目开源了我们 RSS 2025 论文的训练与评估代码，并提供示例数据与模型权重，便于快速复现与二次开发。

主要贡献者： [Jiazhao Zhang](https://jzhzhang.github.io/)、Kunyu Wang、[Shaoan Wang](https://wsakobe.github.io/)、Minghan Li、[Haoran Liu](https://yiconghong.me/)、[Songlin Wei](https://songlin.github.io/)、[Zhongyuan Wang](https://www.wangzhongyuan.com/)、[Zhizheng Zhang](https://scholar.google.com/citations?user=X7M0I8kAAAAJ&hl=en)、[He Wang](https://hughw19.github.io/)<br>

[[论文与附录](https://arxiv.org/pdf/2412.06224)] [[项目主页](https://pku-epic.github.io/Uni-NaVid/)]

![pipeline](./assets/uninavid.png)

## 发布进度
- [x] 训练代码
- [x] 离线评估代码
- [x] 基准评测代码
  - [x] VLN-CE
  - [x] EVT-Bench
- [x] 一小部分 VLN-CE RxR 数据

## 目录
- [安装](#安装)
- [目录结构](#目录结构)
- [准备工作](#准备工作)
  - 模型准备
  - 数据准备
- [训练](#训练)
- [评估](#评估)
  - 离线评估
  - 基准评测
- [引用](#引用)
- [致谢](#致谢)

## 安装

1. 克隆仓库：
```bash
git@github.com:jzhzhang/Uni-NaVid.git
```

2. 安装依赖并以可编辑模式安装本项目：
```bash
conda create -n uninavid python=3.10 -y
conda activate uninavid
cd Uni-NaVid
pip install --upgrade pip  # 启用 PEP 660 支持
pip install -e .
```

3. 安装 flash-attn：
```bash
pip install flash-attn==2.5.9.post1
```

## 目录结构
仓库经过整理，主要目录和脚本如下，便于快速定位功能：
```
Uni-NaVid
├── assets/                     # 资源文件与展示图片
├── rosws/                      # ROS 工作区相关文件
├── scripts/                    # 训练与分布式配置脚本
│   ├── uninavid_stage_1.sh
│   ├── uninavid_stage_2.sh
│   ├── zero2.json
│   └── zero2_offload.json
├── uninavid/                   # 核心代码（模型、训练、处理器等）
│   ├── model/
│   ├── processor/
│   └── train/
├── offline_eval_uninavid.py    # 离线评估脚本
├── api_server.py               # 推理 API 示例
├── pyproject.toml
└── README.md
```
如需跑离线案例，可在项目根目录下放置 `test_cases/` 与示例视频；模型与数据路径可按下方“准备工作”组织。

## 准备工作

### 模型
训练需要下载视觉编码器与语言模型，论文中使用的权重：

| 模型类型 | 模型名称 | 下载链接 |
|------|------|------|
| 编码器 | EVA-CLIP | [ckpt](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth) |
| 预训练模型 | Vicuna-7B | [ckpt](https://huggingface.co/lmsys/vicuna-7b-v1.5) |
| 微调模型 | Uni-NaVid (7B) | [ckpt](https://huggingface.co/Jzzhang/Uni-NaVid/tree/main/uninavid-7b-full-224-video-fps-1-grid-2) |

### 数据

我们提供了论文使用数据的一个小子集，便于快速复现或替换为自有数据，可从 [这里](https://huggingface.co/Jzzhang/Uni-NaVid/tree/main/Nav-Finetune) 下载。数据来自 [VLN-CE](https://github.com/jacobkrantz/VLN-CE) R2R/RxR、[EVT-Bench](https://github.com/wsakobe/TrackVLA)、[ObjectNav](https://arxiv.org/abs/2006.13171) 与 [EQA](https://embodiedqa.org/) 等导航任务。

**注意**：由于授权限制，ObjectNav 的训练未使用 [L3MVN](https://arxiv.org/pdf/2304.05501) 方法，ObjectNav 评测上性能可能略低。

建议的项目目录布局：
```
Uni-NaVid
├── data
│   └── Nav-Finetune
│       ├── nav_videos
│       └── open_uninavid_sampled_500.json
├── model_zoo
│   ├── eva_vit_g.pth
│   ├── <vicuna_weights>          # 可选：从 Vicuna 微调
│   └── <uninavid_weights>
├── scripts
├── uninavid
└── test_cases                    # 可选：离线评估示例
```

## 训练
在 `scripts/uninavid_stage_1.sh` 与 `scripts/uninavid_stage_2.sh` 中设置 `DATA_PATH`、`MODEL_PATH` 为实际数据与模型存放路径。

- 若从 Vicuna-7B 开始微调（需要充足数据）：
```bash
bash scripts/uninavid_stage_1.sh
```
- 若基于发布的 Uni-NaVid 权重继续微调：
```bash
bash scripts/uninavid_stage_2.sh
```

## 评估
评估阶段模型使用在线 token merging（`run_type=eval`），在单张 A100 上推理速度约 5 Hz；配合量化等优化可进一步提速。

### 离线评估
我们提供在真实视频上的离线评估示例，包含 VLN 样例 `vln_1` 与跟踪样例 `tracking_1`。示例视频可从 [这里](https://huggingface.co/Jzzhang/Uni-NaVid/tree/main/test_cases) 下载。
```bash
python offline_eval_uninavid.py test_cases/vln_1 output_dir   # 或 test_cases/tracking_1
```

https://github.com/user-attachments/assets/31592c56-8369-4389-994f-f64b151ebb59

（move to the chair, then turn left and move forward to the humanoid robot and stop.）

https://github.com/user-attachments/assets/5ae851e0-d7fd-4b29-8501-05715febfc47

（follow the man with black top and brown pants.）

### 基准评测
- **VLN-CE** 评测代码见：[NaVid-VLN-CE](https://github.com/jzhzhang/NaVid-VLN-CE)。

| 评测基准 |  TL  |  NE  |  OS  |  SR  |  SPL |
|----------|:----:|:----:|:----:|:----:|:----:|
| Uni-NaVid VLN-CE R2R Val. | 9.22 | 4.96 | 57.4 | 51.8 | 47.7 |
| Uni-NaVid VLN-CE RxR Val. | 18.4 | 5.67 | 66.4 | 56.1 | 44.5 |

- **EVT-Bench** 评测代码见：[TrackVLA](https://github.com/wsakobe/TrackVLA)。

| 评测基准 |  SR  |  TR  |  CR  |
|----------|:----:|:----:|:----:|
| Uni-NaVid EVT-Bench STT | 53.3 | 67.2 | 12.6 |
| Uni-NaVid EVT-Bench STT | 31.9 | 50.1 | 21.3 |
| Uni-NaVid EVT-Bench AT  | 15.8 | 41.5 | 26.5 |

## 引用
如果本工作对您的研究有所帮助，请引用：
```bibtex
@article{zhang2024uni,
    title={Uni-NaVid: A Video-based Vision-Language-Action Model for Unifying Embodied Navigation Tasks},
    author={Zhang, Jiazhao and Wang, Kunyu and Wang, Shaoan and Li, Minghan and Liu, Haoran and Wei, Songlin and Wang, Zhongyuan and Zhang, Zhizheng and Wang, He},
    journal={Robotics: Science and Systems},
    year={2025}
}
```

## 致谢
本项目基于 [LLaMA-VID](https://github.com/dvlab-research/LLaMA-VID) 与 [NaVid](https://github.com/jzhzhang/NaVid-VLN-CE) 进行开发。部分函数已重写以满足开源许可要求。

如有问题，欢迎联系 Jiazhao Zhang：zhngjizh@gmail.com。
