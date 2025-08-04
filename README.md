# Physics-Aware Computer Vision Pipeline using Triton Inference Server

This repository demonstrates a physics-integrated computer vision pipeline deployed on Triton Inference Server. The system detects a basketball in motion, extracts its 2D coordinates using a YOLOv9 model, and sends these coordinates to a Physics-Informed Neural Network (PINN) based on the Navier-Stokes equations. The PINN predicts forces acting in the x and y directions (`u` and `v`) and pressure (`p`), bridging visual and physical insights.

This showcases how vision and physics can jointly contribute to actionable insights in sports — for example, analyzing the forces behind a throw and perfecting athletic techniques.

---

## 📚 Table of Contents

1. [🧠 Physics Model: PhysicsNemo with Navier-Stokes](#-physics-model-physicsnemo-with-navier-stokes)

   * [Training Docker](#training-docker)
   * [Key Highlight](#key-highlight)
   * [Training](#training)
   * [Visualizations](#visualizations)
   * [Export to ONNX](#export-to-onnx-for-deployment)
2. [🧾 Vision Model: YOLOv9](#-vision-model-yolov9)

   * [Triton Deployment](#triton-deployment)
3. [🎥 Demo](#-demo)
4. [✨ Conclusion](#-conclusion)

---

## 🧠 Physics Model: PhysicsNemo with Navier-Stokes

### Training Docker

* **Docker**: `nvcr.io/nvidia/physicsnemo/physicsnemo:25.06`
* **Reference**: [Lid Driven Cavity Flow Example](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/user_guide/basics/lid_driven_cavity_flow.html)

### Key Highlight

Boundary constraints were customized to model the specific physics of a basketball throw.

### Training

Run training using:
[train.py](https://github.com/TejasARathod/PhyCV-PhysicsNemo-Triton-Yolov9/blob/6a84daaab16eb379317291a026a7f1191aa55c24/train.py)

### Visualizations

* Training Stats on TensorBoard:
  ![Training Logs](https://github.com/TejasARathod/PhyCV-PhysicsNemo-Triton-Yolov9/blob/6a84daaab16eb379317291a026a7f1191aa55c24/Tensorboard_TrainingStats.png)

* Flow Predictions (u, v, p fields):
  ![Flow Predictions](https://github.com/TejasARathod/PhyCV-PhysicsNemo-Triton-Yolov9/blob/6a84daaab16eb379317291a026a7f1191aa55c24/Flow_pred_fields.png)

### Export to ONNX for Deployment

Use [convert\_to\_onnx.py](https://github.com/TejasARathod/PhyCV-PhysicsNemo-Triton-Yolov9/blob/6a84daaab16eb379317291a026a7f1191aa55c24/convert_to_onnx.py) to export the trained PyTorch model to ONNX format for inference.

---

## 🧾 Vision Model: YOLOv9

* Trained a small YOLOv9 model
* **Reference**: [YOLOv9 GitHub Repo](https://github.com/WongKinYiu/yolov9.git)

### Triton Deployment

* Based on:

  * [Triton Server YOLO Deployment](https://github.com/levipereira/triton-server-yolo.git)
  * [Triton YOLO Client](https://github.com/levipereira/triton-client-yolo.git)
* **Docker used**: `nvcr.io/nvidia/tritonserver:23.10-py3`

---

## 🎥 Demo

Run the Triton server and launch the client to process a video of a man throwing a basketball. The pipeline:

1. Detects the basketball
2. Extracts `(x, y)` coordinates
3. Predicts `(u, v)` forces and `p` pressure using the PhysicsNemo model

### Demo GIF:

![Demo](https://github.com/TejasARathod/PhyCV-PhysicsNemo-Triton-Yolov9/blob/6a84daaab16eb379317291a026a7f1191aa55c24/Demo.gif)


