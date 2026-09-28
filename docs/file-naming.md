# 文件命名与归档说明

| 原名称 | 整理后名称 | 处理方式 |
| --- | --- | --- |
| `tensorRT_bulid20220423.zip` | `plant-leaf-disease-tensorrt-20220423.zip` | 作为 Release 附件上传，内容与校验和保持不变 |
| ZIP 根目录 `tensorRT_bulid/` | Git 仓库根目录 | 提取源码时移除旧根目录层级；原 ZIP 内部结构不变 |
| `基于TensorRT的植物叶片病害实时检测分类模型优化.pdf` | `docs/plant-leaf-disease-tensorrt-paper.pdf` | 仅改文件名，不修改论文内容 |
| `model_data/cls_classse.txt` | `model_data/leaf_classes.txt` | 修正含糊、拼写不规范的名称，并更新 `ssd.py` 引用 |
| `image (476).JPG` 至 `image (479).JPG` | `samples/leaf-476.jpg` 至 `samples/leaf-479.jpg` | 整理到示例目录，移除空格和括号，并同步更新 Python 引用 |

`ttt.engine`、`test.onnx` 继续保留原运行时名称，位于完整 ZIP 中。代码模块名及历史实验脚本名保持原样，以免改变模块导入关系。

Git 仓库排除 IDE 配置、Python 缓存和超大模型；原始 Release ZIP 完整保留原文件。原始本地 ZIP 与 PDF 也没有被修改。
