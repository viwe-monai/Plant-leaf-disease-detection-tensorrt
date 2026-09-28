#----------------------------------------------------#
#这个文件是需要运行的主要文件
#----------------------------------------------------#
import time
import cv2
import numpy as np
from PIL import Image
from ssd import SSD


if __name__ == "__main__":
    ssd = SSD()
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #----------------------------------------------------------------------------------------------------------#
    mode =  "video"   #"predict"  #"video"
    #-------------------------------------------------------------------------#
    #   crop指定了是否在单张图片预测后对目标进行截取
    #   crop仅在mode='predict'时有效
    #-------------------------------------------------------------------------#
    #----------------------------------------------------------------------------------------------------------#
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0  #0 一般是默认调动的摄像头
    video_fps       = 25.0
    #-------------------------------------------------------------------------#
    #   test_interval用于指定测量fps的时候，图片检测的次数q
    #   理论上test_interval越大，fps越准确。
    #-------------------------------------------------------------------------#
    test_interval = 100
    #-------------------------------------------------------------------------

    if mode == "video":
        capture=cv2.VideoCapture(video_path,cv2.CAP_DSHOW)
        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
        fps = 0.0
        while(True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(ssd.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff

            key = cv2.waitKey(10)        #
            if key & 0xFF == ord('q'):   #
                # 按q退出                 #
                break                    #
            if c==27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()

        cv2.destroyAllWindows()