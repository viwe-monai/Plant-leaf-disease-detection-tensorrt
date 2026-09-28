#不可用



# 导入必用依赖
import tensorrt as trt

import common


###create trt engine from parsering onnx model
TRT_LOGGER = trt.Logger() #初始化一个消息模块，用来存放发布消息尤其是警告报错
onnx_file_path = 'test1.onnx' #onnx模型路径
engine_file_path = './sample.trt' #目标trt文件保存模型路径

with trt.Builder(TRT_LOGGER) as builder,\
 builder.create_network(common.EXPLICIT_BATCH) as network, \
builder.create_builder_config() as config, \
trt.OnnxParser(network, TRT_LOGGER) as parser:
    config.max_workspace_size = 1 << 28 # 256MiB
    builder.max_batch_size = 1
     # Parse model file
    print('Loading ONNX file from path {}...'.format(onnx_file_path))
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read()) #关键句：读取并解析模型

    network.get_input(0).shape = [1, 3, 224, 224]
    #固定我们预先设定的模型输出shape,这里是图像分类模型，所以输入形状为1，3，224，224
    print('Completed parsing of ONNX file')
    print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
    #engine = builder.build_engine(network, config)#根据解析与配置文件创建engine
    engine = builder.build_serialized_network(network, config)
    print("Completed creating Engine")
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())      #序列化并写入文件

