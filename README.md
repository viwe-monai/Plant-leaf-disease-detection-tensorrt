# 植物叶片病害检测与 TensorRT 推理优化

与论文《基于 TensorRT 的植物叶片病害实时检测分类模型优化》相关的研究代码与模型归档。系统先通过 **SSD + MobileNetV2** 定位叶片，再对叶片区域进行 **ResNet-152、33 类健康/病害分类**，使用 TensorRT 加速分类阶段，并通过 OpenCV 显示视频检测结果。

本仓库提供整理后的可浏览源码、SSD 权重、示例图片和论文。完整原始 ZIP 放在 Release 附件中，包含体积较大的 ONNX 与 TensorRT 引擎。代码属于历史研究版本，仍需适配运行环境，未在此次整理中重新执行模型推理。

## 下载

- [阅读论文 PDF](docs/plant-leaf-disease-tensorrt-paper.pdf)
- [下载完整原始项目 ZIP（约 576.82 MiB）](https://github.com/viwe-monai/Plant-leaf-disease-detection-tensorrt/releases/download/archive-20220423/plant-leaf-disease-tensorrt-20220423.zip)
- [查看归档 Release](https://github.com/viwe-monai/Plant-leaf-disease-detection-tensorrt/releases/tag/archive-20220423)

ZIP 的内容保持原样，仅规范了下载文件名。仓库中的源码则整理了部分文件名并同步更新引用；详细对应关系见 [文件命名说明](docs/file-naming.md)。原始 ZIP、`test.onnx` 和 `ttt.engine` 超过普通 Git 单文件大小限制，因此不直接放入 Git 历史。

## 流程

```text
摄像头 / 图像
    → SSD + MobileNetV2 定位叶片
    → 裁剪叶片区域
    → ResNet-152 分类器 / TensorRT 引擎
    → 33 类标签与分数
    → OpenCV 显示
```

检测与分类是两个模型：`ep100-loss2.500-val_loss1.787.pth` 为 SSD 检测模型；分类器的原始 PyTorch checkpoint 不在这份归档内，不能用 SSD 权重代替。

## 目录

| 路径 | 用途 |
| --- | --- |
| `predict.py` | 主要入口，当前启用摄像头视频模式 |
| `ssd.py` | SSD 检测及裁剪区域的病害分类调用 |
| `nets/` | SSD、MobileNetV2、VGG 与相关训练组件 |
| `utils/` | 预处理、先验框、后处理、训练和评估辅助函数 |
| `predict_use_in_SSD.py` | 供 SSD 流程调用的 TensorRT 分类函数 |
| `predict_trt_well_clean.py` | TensorRT 单图分类示例 |
| `predict_test_success_20220423.py` | 保留的分类实验脚本 |
| `DiseaseDetection.py`、`DiseaseDetection_pth.py` | PyTorch 分类器与对比实验代码，需要另备分类 checkpoint |
| `resnet_pth_to_onnx.py` | ResNet 分类器导出 ONNX 的历史脚本 |
| `change.py`、`common.py` | TensorRT 构建实验及辅助代码；`change.py` 原文件标注为不可用 |
| `ep100-loss2.500-val_loss1.787.pth` | 随附 SSD 权重，约 14.24 MiB |
| `model_data/leaf_classes.txt` | SSD 检测类别，当前为 `leaves` |
| `model_data/simhei.ttf` | 原项目附带的中文绘制字体 |
| `cat_to_name.json` | 原始分类标签映射资料 |
| `samples/` | 4 张原始示例图像，已统一为 `leaf-NNN.jpg` 命名 |
| `docs/` | 论文、命名对应表与原始 ZIP 校验和 |

归档还包含 `test.onnx`（237,558,262 字节）和 `ttt.engine`（392,393,551 字节）。运行前按下文从 ZIP 中取出。

## 环境依赖

需要 NVIDIA GPU、兼容的驱动/CUDA/TensorRT 环境，以及 Python、PyTorch、torchvision、NumPy、OpenCV、Pillow 和 PyCUDA。训练或评估辅助代码还使用 tqdm、SciPy、Matplotlib、TensorBoard 等包；ONNX 导出需要对应的 PyTorch/ONNX 支持。

原始项目没有提供可复现的版本锁定文件。代码使用 `engine.max_batch_size`、binding API 和 `context.execute_async` 等旧式 TensorRT 接口，以及 Pillow 的 `Image.ANTIALIAS`，不能假定直接兼容任意新版依赖。请根据原始引擎、GPU 和目标设备选择匹配环境，或先完成 API 迁移。

论文包含主机与 Jetson Nano 实验，但当前 `predict.py` 的摄像头入口指定了 Windows 的 `cv2.CAP_DSHOW`。Linux / Jetson 部署时应改用适合设备的摄像头后端或 GStreamer 管线。

## 准备模型

克隆仓库后，先下载完整归档，再从中取出两个大模型到仓库根目录：

```bash
git clone https://github.com/viwe-monai/Plant-leaf-disease-detection-tensorrt.git
cd Plant-leaf-disease-detection-tensorrt

curl -L --fail -o plant-leaf-disease-tensorrt-20220423.zip \
  https://github.com/viwe-monai/Plant-leaf-disease-detection-tensorrt/releases/download/archive-20220423/plant-leaf-disease-tensorrt-20220423.zip

sha256sum -c docs/archive.sha256
unzip -j plant-leaf-disease-tensorrt-20220423.zip \
  tensorRT_bulid/ttt.engine tensorRT_bulid/test.onnx
```

也可通过解压软件手动复制。不要直接用归档源码覆盖仓库，避免覆盖整理后的文件名与引用。

TensorRT 序列化引擎与生成时的硬件和软件环境有关。如果 `ttt.engine` 无法反序列化，需要基于 `test.onnx` 在匹配的目标环境中重新构建，并核对输入输出与当前推理代码的接口约定。

## 运行示例

完成依赖和模型准备后，在仓库根目录运行单图分类：

```bash
python predict_trt_well_clean.py
```

默认读取 `samples/leaf-479.jpg` 和 `ttt.engine`，输入按 `1 × 3 × 224 × 224` 构造，输出按 33 类处理。更换输入时修改脚本中的 `filename`；当前没有统一的命令行参数接口。

运行完整摄像头检测流程：

```bash
python predict.py
```

- 默认摄像头索引为 `0`；Windows 使用 `CAP_DSHOW`，其他平台需要调整。
- SSD 当前输入尺寸为 `512 × 512`，检测阈值为 `0.6`、NMS IoU 为 `0.45`，配置位于 `ssd.py`。
- 代码默认启用 CUDA，分类还依赖 TensorRT / PyCUDA，因此不是纯 CPU 示例。
- 窗口中按 `q` 或 Esc 可退出。源文件注释提到 `predict` 模式，但当前 `predict.py` 仅实现了 `video` 分支；不能只改字符串就获得交互式单图检测功能。

## 复现实验与已知限制

- `LeavesDiseaseClassification220327.pth` **不在原始 ZIP 内**。`DiseaseDetection.py`、`DiseaseDetection_pth.py` 和 ONNX 导出脚本引用它，运行这些入口前需另行准备，并统一脚本中根目录与 `model_data/` 两种路径写法。
- `resnet_pth_to_onnx.py` 导出名为 `test1.onnx`，归档自带的文件名为 `test.onnx`；两者不能仅凭文件名认定为相同模型。
- `change.py` 标注为不可用，其构建返回值与后续序列化调用需按实际 TensorRT 版本修正，不应直接当作已经验证的构建命令。
- 分类入口之间存在预处理差异：单图 TensorRT 示例直接缩放到 224 × 224，而其他入口采用缩略及裁剪。比较结果前应对齐预处理、标签顺序和模型。
- `predict_use_in_SSD.py` 当前会在每次分类调用中加载引擎和创建执行上下文，尚未整理为可复用的推理服务。
- 目录中虽有训练辅助组件，但没有完整训练入口、训练数据和所有原始 checkpoint，不能据此保证完整复现论文训练过程。
- 本次整理仅检查了文件、源码与论文，未验证推理精度、速度或跨设备兼容性。

## 论文

徐泽华，李坚孝，邓树源，吴家隐，高嘉晖，潘明毅．**基于 TensorRT 的植物叶片病害实时检测分类模型优化**．计算机系统应用，2023，32(2)：94–101．DOI：[10.15888/j.cnki.csa.008977](https://doi.org/10.15888/j.cnki.csa.008977)。

论文报告了其特定实验设置下的分类模型与完整系统加速效果。这些是论文中的实验结果，并非对本次归档代码重新测试所得的性能承诺。

论文 PDF 保持原内容，仅调整仓库文件名；代码、模型、字体、第三方组件和论文保留各自原有版权。本次整理没有为整个项目新增开源许可证。
