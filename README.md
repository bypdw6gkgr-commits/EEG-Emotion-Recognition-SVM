EEG Emotion Recognition with SVM (5-Class)

基于 EEGEmotions-27 数据集的 5 分类脑电情感识别系统

📖 项目简介

本项目是一个基于 Python 的脑机接口（BCI）情感识别系统。它利用 SVM 算法对 EEG 信号进行解码，能够识别 5 种核心情绪：Joy (愉悦), Anger (愤怒), Fear (害怕), Calmness (平静), Sadness (悲伤)。

项目包含完整的端到端流水线：

预处理：动态时间截取（去头5秒）、抗噪滤波（1-40Hz）。

特征工程：提取 PSD 频域能量 + DASM 左右脑不对称特征。

数据增强：使用滑动窗口（Sliding Window）增加样本量。

模型训练：基于 RBF 核的 SVM 分类器，集成抗噪均衡策略。

预测系统：具备置信度阈值机制，能够对未知情绪进行拒识。

📂 数据集说明 (Dataset)

本项目使用 EEGEmotions-27 数据集。
⚠️ 注意：本仓库不包含原始数据集文件。

如果您想运行本项目，请前往原作者仓库或官方渠道下载数据：

Dataset Name: EEGEmotions-27

Source Paper: EEGEmotions-27: A Large-Scale EEG Dataset Annotated With 27 Fine-Grained Emotion Labels (Phuong et al., 2025)

数据准备步骤：

下载原始数据集。

将所有 .txt 格式的 EEG 原始文件放入项目根目录下的 eeg_raw/ 文件夹中。

(可选) 创建 test_data/ 文件夹并放入部分文件用于测试预测功能。

🛠️ 环境依赖

请确保安装以下 Python 库：

pip install numpy mne scikit-learn scipy joblib matplotlib


🚀 快速开始

1. 训练模型

运行训练脚本，自动从 eeg_raw/ 读取数据并训练模型：

python train_model_final_5class.py


成功后会生成 bci_emotion_model.pkl 文件。

2. 预测新数据

运行预测脚本，对 test_data/ 中的文件进行情感分析：

python predict_new_data_final.py


📊 引用 (Citation)

本项目使用了 Phuong 等人发布的数据集，特此致谢。

@ARTICLE{11168214,
  author={Phuong, Huy-Tung and Im, Eun-Tack and Oh, Myeong-Seok and Gim, Gwang-Yong},
  journal={IEEE Access}, 
  title={EEGEmotions-27: A Large-Scale EEG Dataset Annotated With 27 Fine-Grained Emotion Labels}, 
  year={2025},
  volume={13},
  pages={176915-176932},
  doi={10.1109/ACCESS.2025.3620677}
}


📝 License

MIT License


