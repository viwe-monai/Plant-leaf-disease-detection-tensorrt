#成功运行trt分类模型进行病害识别，准确率与网上的web上的差不多，虽然精度对下降

import pycuda.autoinit   #这个包不能删除，会报错
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import time
from PIL import Image
#filename = 'samples/leaf-479.jpg'
max_batch_size = 1


#标签字典
class_to_idx = {'Apple___Apple_scab': 0, 'Apple___Black_rot': 1, 'Apple___Cedar_apple_rust': 2, 'Apple___healthy': 3, 'Cherry___Powdery_mildew': 4, 'Cherry___healthy': 5, 'Corn___Cercospora_leaf_spot Gray_leaf_spot': 6, 'Corn___Common_rust': 7, 'Corn___Northern_Leaf_Blight': 8, 'Corn___healthy': 9, 'Grape___Black_rot': 10, 'Grape___Esca_(Black_Measles)': 11, 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 12, 'Grape___healthy': 13, 'Peach___Bacterial_spot': 14, 'Peach___healthy': 15, 'Pepper,_bell___Bacterial_spot': 16, 'Pepper,_bell___healthy': 17, 'Potato___Early_blight': 18, 'Potato___Late_blight': 19, 'Potato___healthy': 20, 'Strawberry___Leaf_scorch': 21, 'Strawberry___healthy': 22, 'Tomato___Bacterial_spot': 23, 'Tomato___Early_blight': 24, 'Tomato___Late_blight': 25, 'Tomato___Leaf_Mold': 26, 'Tomato___Septoria_leaf_spot': 27, 'Tomato___Spider_mites Two-spotted_spider_mite': 28, 'Tomato___Target_Spot': 29, 'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 30, 'Tomato___Tomato_mosaic_virus': 31, 'Tomato___healthy': 32}
idx_to_class = {v: k for k, v in class_to_idx.items()}

TRT_LOGGER = trt.Logger()  # This logger is required to build an engine

def softmax(x):
    x_exp = np.exp(x)
    #如果是列向量，则axis=0
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    s = x_exp / x_sum
    return s


#对需要预测的图像进行正则化处理
#调用其他的正则化处理方式
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Process a PIL image for use in a PyTorch model

    size = 256, 256
    image.thumbnail(size, Image.ANTIALIAS)
    image = image.crop((128 - 112, 128 - 112, 128 + 112, 128 + 112))
    npImage = np.array(image)
    npImage = npImage / 255.

    imgA = npImage[:, :, 0]
    imgB = npImage[:, :, 1]
    imgC = npImage[:, :, 2]

    imgA = (imgA - 0.485) / (0.229)
    imgB = (imgB - 0.456) / (0.224)
    imgC = (imgC - 0.406) / (0.225)

    npImage[:, :, 0] = imgA
    npImage[:, :, 1] = imgB
    npImage[:, :, 2] = imgC

    npImage = np.transpose(npImage, (2, 0, 1))

    return npImage


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
           在这种情况下，host_mom表示cpu内存，而device表示GPU内存
        """
        self.host = host_mem
        self.device = device_mem
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        # 分配主机和设备缓冲区
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        # 将设备缓冲区追加到设备绑定。
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        # 添加到适当的列表。
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def get_engine( engine_file_path=""):
        # 加载一个序列化引擎
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    # 将数据从CPU传输到GPU
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    # 运行推理，进行预测
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    # 将预测信息从GPU转移回来
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    # 同步流
    stream.synchronize()
    # Return only the host outputs.
    # 只返回主机输出
    return [out.host for out in outputs]

def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs

trt_engine_path = 'ttt.engine'

# 从本地得到engine
def predict(filename):
    engine = get_engine(trt_engine_path)
    # Create the context for this engine
    # 为这个引擎创建上下文
    context = engine.create_execution_context()
    # Allocate buffers for input and output
    # 为输入和输出分配缓冲区
    inputs, outputs, bindings, stream = allocate_buffers(engine) # input, output: host # bindings
    start = time.time()

########## Do inference
# 进行预测

    t1 = time.time()
    #img_np_nchw = get_img_np_nchw(filename)
    img_np_nchw = process_image(filename)
    img_np_nchw = img_np_nchw.astype(dtype=np.float32)
    shape_of_output = (max_batch_size, 33)  #33为分类的数量
    # Load data to the buffer
    # 将数据加载到缓冲区
    inputs[0].host = img_np_nchw.reshape(-1)
    # inputs[1].host = ... for multiple input
    #t1 = time.time()
    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream) # numpy data
    feat = postprocess_the_outputs(trt_outputs[0], shape_of_output)
    result = softmax(feat)
    score, index = np.max(result, axis=1), np.argmax(result, axis=1)
    top_class=idx_to_class[index[0]]#序号转标签
    t2 = time.time()
    print(score[0], top_class)
    print('used time: ', t2 - t1)
    return  score[0], top_class