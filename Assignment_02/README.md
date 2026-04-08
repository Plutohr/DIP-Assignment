# Assignment 2: DIP with PyTorch

  本次作业分为两部分：

  1. Traditional DIP: Poisson Image Editing
  2. Deep Learning DIP: Pix2Pix

  ---

  ## 1. Environment

  本作业在 `WSL + Python 3.12 + PyTorch` 环境下完成，训练与验证过程中使用 `NVIDIA GeForce RTX 3050 Laptop GPU`。

  ### Main Dependencies

  - `torch 2.11.0+cu126`
  - `torchvision 0.26.0+cu126`
  - `numpy`
  - `pillow`
  - `opencv-python`
  - `gradio`

  ---

  ## 2. Part 1: Poisson Image Editing

  ### 2.1 Task

  根据作业要求，补全 `run_blending_gradio.py` 中的两个部分：

  - `create_mask_from_points(...)`
  - `cal_laplacian_loss(...)`

  ### 2.2 Implementation

  #### Polygon to Mask

  将用户在前景图像中选取的 polygon 区域转换为二值 mask：

  - polygon 外部为 `0`
  - polygon 内部为 `255`

  #### Laplacian Loss

  使用 `torch.nn.functional.conv2d` 对图像做拉普拉斯卷积，并在 mask 区域内比较前景图与融合图的梯度信息，通过梯度下降不断
  优化融合结果。

  ### 2.3 Interactive UI

  作业提供了 Gradio 交互界面：

  - 上传前景图与背景图
  - 在前景图中手动点击 polygon 区域
  - 调整 `dx / dy`
  - 点击 `Blend Images` 执行融合

  运行方式：

  ```bash
  cd Assignments/02_DIPwithPyTorch
  source .venv/bin/activate
  python run_blending_gradio.py

  ### 2.4 Experiment Results
  为了更直观地展示交互式 Poisson Image Editing 的使用流程与融合效果，这里给出一个 demo 示例。该示例展示了从前景区域选
  择、位置调整到最终融合结果的完整过程。
  poisson demo(pics/demo.gif)

  ———

  ## 3. Part 2: Pix2Pix

  ### 3.1 Task

  根据作业要求，补全 Pix2Pix/FCN_network.py 中的 Fully Convolutional Network。

  ### 3.2 Network Design

  本次实现了一个简单的 encoder-decoder 全卷积网络：

  - Encoder: Conv2d + BatchNorm2d + ReLU
  - Decoder: ConvTranspose2d + BatchNorm2d + ReLU
  - Output Layer: Tanh

  输入图像尺寸为 256 x 256，输出为对应的三通道图像。

  ### 3.3 Dataset

  本次实验首先使用作业提供的 facades 数据集进行训练。

  下载命令：

  cd Assignments/02_DIPwithPyTorch/Pix2Pix
  bash download_facades_dataset.sh

  数据划分如下：

  - Training set: 400 images
  - Validation set: 100 images

  ### 3.4 Training Configuration

  训练命令：

  cd Assignments/02_DIPwithPyTorch/Pix2Pix
  source ../.venv/bin/activate
  python train.py

  主要训练配置：

  - Optimizer: Adam
  - Learning rate: 0.001
  - Betas: (0.5, 0.999)
  - Loss: L1Loss
  - Epochs: 300
  - Batch size: 100

  ### 3.5 Training Process

  训练过程正常收敛。前期 loss 下降较快，后期进入平台期。

  部分日志如下：

  - Epoch 1 validation loss: 0.8034
  - Epoch 10 validation loss: 0.5431
  - Epoch 50 validation loss: 0.4612
  - Epoch 100 validation loss: 0.4358
  - Epoch 200 validation loss: 0.4469
  - Epoch 300 validation loss: 0.4468


  ## 5. File Structure

  Assignments/02_DIPwithPyTorch
  ├── README.md
  ├── requirements.txt
  ├── SETUP.md
  ├── run_blending_gradio.py
  ├── pics
  │   └── demo.png
  ├── data_poisson
  │   ├── water
  │   ├── monolisa
  │   └── equation
  └── Pix2Pix
      ├── README.md
      ├── FCN_network.py
      ├── facades_dataset.py
      ├── train.py
      └──download_facades_dataset.sh

  ———

  ## 6. References

  - Poisson Image Editing (https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf)
  - Pix2Pix Project Page (https://phillipi.github.io/pix2pix/)
  - Fully Convolutional Networks for Semantic Segmentation (https://arxiv.org/abs/1411.4038)
  - PyTorch Documentation (https://pytorch.org/)
