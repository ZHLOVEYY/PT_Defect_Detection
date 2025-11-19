from ultralytics import YOLO

if __name__ == '__main__':
    # 加载一个预训练的 YOLOv8n 模型
    # 'n' 代表 nano，是最小最快的版本，适合入门
    model = YOLO('yolov8n.pt')

    # 开始训练
    # data: 指向我们刚刚创建的数据集配置文件
    # epochs: 训练轮次，所有图片被训练一次为一个 epoch。可以先从 50 开始尝试。
    # imgsz: 输入图像的尺寸
    # device: 在 macOS 上，可以留空或设置为 'mps' 来使用 Apple Silicon GPU
    results = model.train(
        data='powergrid_data.yaml',
        epochs=50,
        imgsz=640,
        device='cpu',
        lr0=0.001  # 新增：将初始学习率降低10倍
    )

    # 训练完成后，结果会自动保存在 'runs/detect/trainX' 文件夹中
    print("训练完成！结果保存在 runs 文件夹下。")