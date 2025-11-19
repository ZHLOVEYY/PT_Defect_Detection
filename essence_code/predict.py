from ultralytics import YOLO
from PIL import Image

# 加载你训练好的模型
# 注意：路径要指向你训练轮次中表现最好的权重文件
model = YOLO('runs/detect/train5/weights/best.pt')

# 指定你要检测的图片或视频源
# 可以是单张图片、视频文件、或者一个包含多张图片的文件夹
source = 'testpic/test1.jpg'  

# 执行推理
results = model.predict(source, save=True, imgsz=640, conf=0.5)

# `save=True` 会将带有标注框的结果图片保存在 `runs/detect/predictX` 文件夹下
# `conf=0.5` 是置信度阈值，只显示预测置信度高于50%的目标

print("推理完成！结果已保存。")