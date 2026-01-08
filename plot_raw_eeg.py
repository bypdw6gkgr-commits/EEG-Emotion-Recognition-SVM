import os
import numpy as np
import mne
import matplotlib.pyplot as plt
import random

# ==========================================
# 配置区域 (请在此处修改时间区间)
# ==========================================
DATA_PATH = 'eeg_raw' 
CH_NAMES = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 
            'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
SFREQ = 256 

# 【核心修改点】在此处设置你想要查看的时间段
START_TIME = 1.0    # 起始时间 (秒) -> 设置为 5.0 以跳过前 5 秒的视频介绍
DURATION = 15.0     # 持续时长 (秒) -> 设置为 10.0 以查看接下来 10 秒的反应

# ==========================================
# 核心函数
# ==========================================

def load_txt_to_mne(file_path):
    """读取txt并转换为MNE对象"""
    try:
        data = np.loadtxt(file_path)
        if data.shape[0] != 14: data = data.T
        if data.shape[0] != 14: return None
        
        data = data * 1e-6 # uV -> V
        
        info = mne.create_info(ch_names=CH_NAMES, sfreq=SFREQ, ch_types='eeg')
        raw = mne.io.RawArray(data, info, verbose=False)
        return raw
    except Exception as e:
        print(f"读取失败: {e}")
        return None

def plot_raw_waves(raw, filename):
    """使用 Matplotlib 绘制指定时间段的堆叠波形图"""
    data = raw.get_data() # (n_channels, n_samples)
    times = raw.times
    
    # --- 计算文件信息 ---
    total_samples = data.shape[1]
    total_duration = total_samples / SFREQ
    
    print(f"\n>>> 文件分析: {filename}")
    print(f"    - 总时长: {total_duration:.2f} 秒")
    print(f"    - 目标区间: 从 {START_TIME} 秒 到 {START_TIME + DURATION} 秒")
    print("-" * 30)
    
    # --- 【关键逻辑】根据 START_TIME 和 DURATION 截取数据 ---
    start_sample = int(START_TIME * SFREQ)
    end_sample = int((START_TIME + DURATION) * SFREQ)
    
    # 边界检查 1: 起始时间超过总时长
    if start_sample >= total_samples:
        print(f"错误: 起始时间 ({START_TIME}s) 超过了文件总时长 ({total_duration:.2f}s)")
        print("请尝试减小 START_TIME。")
        return

    # 边界检查 2: 结束时间超过总时长 (自动截断)
    if end_sample > total_samples:
        print(f"提示: 请求的结束时间超出文件范围，将显示到文件末尾 ({total_duration:.2f}s)。")
        end_sample = total_samples
    
    # 执行截取
    plot_data = data[:, start_sample:end_sample]
    plot_times = times[start_sample:end_sample]
    
    if plot_data.shape[1] == 0:
        print("错误: 选定区间内没有数据点。")
        return

    # --- 开始绘图 ---
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 计算偏移量 (Offset)
    step = np.std(plot_data) * 5
    if step == 0: step = 1e-5
    
    print(f"正在绘制波形 ...")
    
    for i, ch_name in enumerate(CH_NAMES):
        offset = i * step
        ax.plot(plot_times, plot_data[i] + offset, linewidth=1, label=ch_name)
        # 在左侧标注通道名
        ax.text(plot_times[0], offset, ch_name, 
                verticalalignment='center', fontweight='bold', fontsize=10, color='#333')

    ax.set_title(f"EEG Raw Data | File: {filename}\nTime Window: {START_TIME}s - {plot_times[-1]:.2f}s", fontsize=14)
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_yticks([]) 
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 主程序
# ==========================================
def main():
    if not os.path.exists(DATA_PATH):
        print(f"错误: 找不到文件夹 {DATA_PATH}")
        return

    files = [f for f in os.listdir(DATA_PATH) if f.endswith('.txt')]
    
    if not files:
        print(f"错误: {DATA_PATH} 文件夹里没有 .txt 文件")
        return

    # 随机选一个文件
    #target_file = random.choice(files)
    # 如果你想指定文件，取消下面这行的注释并修改文件名
    target_file = "1_1.0.txt"
    
    print(f"随机选中文件: {target_file}")
    
    file_path = os.path.join(DATA_PATH, target_file)
    raw = load_txt_to_mne(file_path)
    
    if raw:
        # 滤波 (可选)
        raw.filter(l_freq=1.0, h_freq=40.0, verbose=False)
        plot_raw_waves(raw, target_file)
    else:
        print("文件读取失败。")

if __name__ == "__main__":
    main()