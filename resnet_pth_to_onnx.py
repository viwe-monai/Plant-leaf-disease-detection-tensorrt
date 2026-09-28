#将pth模型装换成onnx模型

import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from collections import OrderedDict
device = torch.device("cpu" )  #这里使用GPU不太行

#model = torch.load("LeavesDiseaseClassification220327.pth") # pytorch模型加载

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath,map_location=device)#CNN加载模型使用GPU
    model = models.resnet152()

    # Our input_size matches the in_features of pretrained model
    input_size = 2048
    output_size = 33      #33是植物病害分类的数量

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(2048, 512)),
        ('relu', nn.ReLU()),
        # ('dropout1', nn.Dropout(p=0.2)),
        ('fc2', nn.Linear(512, 33)),   #33是植物病害分类的数量
        ('output', nn.LogSoftmax(dim=1))
    ]))

    # Replacing the pretrained model classifier with our classifier
    model.fc = classifier

    model.load_state_dict(checkpoint['state_dict'])

    return model, checkpoint['class_to_idx']


model, class_to_idx = load_checkpoint('LeavesDiseaseClassification220327.pth')



batch_size = 1  #批处理大小
input_shape = (3, 224, 224)   #输入数据,改成自己的输入shape

# #set the model to inference mode
model.eval()

x = torch.randn(batch_size, *input_shape)   # 生成张量
x = x.to(device)
export_onnx_file = "test1.onnx"		# 目的ONNX文件名
torch.onnx.export(model,
                    x,
                    export_onnx_file,
                    opset_version=10,
                    do_constant_folding=True,	# 是否执行常量折叠优化
                    input_names=["input"],	# 输入名
                    output_names=["output"],	# 输出名
                    dynamic_axes={"input":{0:"batch_size"},  # 批处理变量
                                    "output":{0:"batch_size"}})



