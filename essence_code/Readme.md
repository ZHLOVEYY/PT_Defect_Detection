# YOLOv8-based Intelligent Detection System for Power Equipment Thermal Imaging

## 1. Project Background

The safe and stable operation of the power grid is of paramount importance. Key equipment such as voltage transformers (PTs) and surge arresters are critical components. Due to factors like material degradation and electrical breakdown, this equipment can exhibit abnormal heating. This overheating is a precursor to severe failures like equipment breakdown, and if not detected and addressed in time, it can seriously threaten the safety of the power grid.

This project aims to build an intelligent analysis system using deep learning and computer vision techniques. By automatically detecting thermal defects in power equipment from infrared thermal images, the system provides early warnings for potential faults.

## 2. Project Value

- **Promoting Intelligent Power Grid O&M**: Empowering the traditional power industry with AI, providing key technical support, and enhancing the overall competitiveness of the sector.
- **Building an Intelligent Analysis System**: Automating the workflow from on-site image capture and intelligent analysis to remote querying. This enables timely discovery of hidden dangers, reducing manual inspection costs and economic losses.
- **Driving Power Industry Upgrades**: Deeply integrating infrared thermal imaging with artificial intelligence, expanding the application boundaries of related technologies, and ensuring the reliability of the energy supply.

## 3. Technical Implementation

This project is based on the **Ultralytics YOLOv8** object detection framework. It uses a custom dataset of infrared thermal images to train the model for accurately identifying abnormal heating areas in power equipment.

- **Model**: YOLOv8n (nano version)
- **Framework**: PyTorch
- **Key Dependencies**: `ultralytics`, `Pillow`

## 4. Project Structure

```
essence_code/
├── Dataset/            # Dataset folder (not uploaded)
│   ├── train/
│   ├── valid/
│   └── test/
├── runs/               # Training and detection results (ignored by .gitignore)
├── testpic/            # Images for testing
│   └── test1.jpg
├── powergrid_data.yaml # Dataset configuration file
├── train.py            # Training script
├── predict.py          # Inference/prediction script
├── test.py             # Test script (optional)
└── yolov8n.pt          # YOLOv8n pre-trained weights
```

**Note**: Due to data confidentiality, the dataset used in this project has not been uploaded to the repository. The `powergrid_data.yaml` file defines the dataset structure and classes, but users need to prepare their own image and label files.

## 5. Installation and Usage

### 5.1. Environment Setup

1.  **Clone the Project**
    ```bash
    git clone <your-repository-url>
    cd newstart
    ```

2.  **Create and Activate Conda Environment** (Recommended)
    ```bash
    conda create --name yolo_env python=3.10
    conda activate yolo_env
    ```

3.  **Install Dependencies**
    The main dependency is `ultralytics`, which will automatically install PyTorch and other related libraries.
    ```bash
    pip install ultralytics
    ```
    If you encounter network issues, you can try using a different package mirror.

### 5.2. Model Training

1.  **Prepare the Dataset**
    Please prepare your dataset according to the format specified in `powergrid_data.yaml` and place it in the `Dataset/` directory.

2.  **Start Training**
    Run the `train.py` script to start training. You can adjust parameters within the script as needed, such as `epochs`, `imgsz`, `lr0`, etc.
    ```bash
    python train.py
    ```
    The trained model weights and results will be saved in the `runs/detect/trainX/` directory.

### 5.3. Model Inference

1.  **Update Model Path**
    Open the `predict.py` file and change the path in `model = YOLO(...)` to your best-trained weights file (e.g., `'runs/detect/train/weights/best.pt'`).

2.  **Run Prediction**
    Execute the `predict.py` script to perform detection on images in the `testpic/` folder.
    ```bash
    python predict.py
    ```
    The detection results (images with bounding boxes) will be saved in the `runs/detect/predictX/` directory.

## 6. Optimization and Future Work

This project demonstrates the feasibility of using YOLOv8 for thermal defect detection in power equipment. Future work can be expanded in the following areas:

- **Model Optimization**:
    - Experiment with larger YOLOv8 models (e.g., YOLOv8s/m/l) to achieve higher accuracy.
    - Adjust the network architecture of the YOLO model, such as modifying convolutional layer parameters.
- **Hyperparameter Tuning**:
    - Experiment with and fine-tune key hyperparameters like learning rate, number of epochs, and input image size (imgsz).
- **Data and Feature Engineering**:
    - Expand the dataset with more samples from complex scenarios.
    - Explore methods like vision-language models to incorporate multi-dimensional information.
- **Model Deployment**:
    - Deploy the trained model to edge computing devices (e.g., Jetson Nano, Raspberry Pi) for real-time, on-site detection.
- **Functional Expansion**:
    - Integrate temperature data to move from defect detection to quantitative analysis of specific temperature readings.

