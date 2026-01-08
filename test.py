import os
import numpy as np
import mne
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.signal import welch

# ==========================================
# 1. 配置区域 (必须与训练时完全一致)
# ==========================================
TEST_DATA_PATH = 'test_data'       # 测试数据文件夹
MODEL_FILE = 'bci_5class_model_boost.pkl' # 你的增强版模型文件名

CH_NAMES = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 
            'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
SFREQ = 256 

# 5 种核心情绪
TARGET_EMOTIONS = {
    20: "Joy (愉悦)",
    5:  "Anger (愤怒)",
    17: "Fear (害怕)",
    10: "Calmness (平静)",
    24: "Sadness (悲伤)"
}

# 左右脑对称电极对 (必须与训练代码一致)
ASYMMETRY_PAIRS = [(0, 13), (1, 12), (2, 11), (3, 10), (4, 9), (5, 8), (6, 7)]

# 滑动窗口配置 (用于预测时的切片)
WINDOW_SIZE = 2.0  
WINDOW_STRIDE = 1.0 

# ==========================================
# 2. 核心处理流水线
# ==========================================

def load_txt_to_mne(file_path):
    try:
        data = np.loadtxt(file_path)
        if data.shape[0] != 14: data = data.T
        if data.shape[0] != 14: return None
        data = data * 1e-6 
        info = mne.create_info(ch_names=CH_NAMES, sfreq=SFREQ, ch_types='eeg')
        raw = mne.io.RawArray(data, info, verbose=False)
        return raw
    except: return None

def extract_features_from_epochs(epochs_data):
    """
    输入: (n_epochs, n_channels, n_times)
    输出: (n_epochs, n_features)
    """
    n_epochs, n_channels, n_times = epochs_data.shape
    freqs, psds = welch(epochs_data, fs=SFREQ, nperseg=SFREQ, axis=-1)
    freq_bands = {'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 45)}
    
    feats_list = []
    for i in range(n_epochs):
        epoch_feats = []
        psd = psds[i]
        
        channel_band_powers = np.zeros((n_channels, len(freq_bands)))
        for b_i, (band_name, (fmin, fmax)) in enumerate(freq_bands.items()):
            mask = (freqs >= fmin) & (freqs <= fmax)
            band_power = np.mean(psd[:, mask], axis=1)
            channel_band_powers[:, b_i] = band_power
            epoch_feats.extend(np.log10(band_power + 1e-10))
        
        for (l_idx, r_idx) in ASYMMETRY_PAIRS:
            for b_i in range(len(freq_bands)):
                left_pow = channel_band_powers[l_idx, b_i]
                right_pow = channel_band_powers[r_idx, b_i]
                diff = np.log10(left_pow + 1e-10) - np.log10(right_pow + 1e-10)
                epoch_feats.append(diff)
                
        feats_list.append(epoch_feats)
    return np.array(feats_list)

def process_file_and_predict(file_path, model):
    """
    读取文件 -> 切片 -> 批量预测 -> 投票
    """
    raw = load_txt_to_mne(file_path)
    if raw is None: return None, 0
    
    # 1. 预处理
    raw.set_eeg_reference('average', projection=False, verbose=False)
    raw.filter(l_freq=1.0, h_freq=45.0, verbose=False)
    
    # 2. 切片 (生成多个 Epochs)
    # 如果文件太短不够切片，就只切一个
    try:
        epochs = mne.make_fixed_length_epochs(raw, duration=WINDOW_SIZE, overlap=WINDOW_SIZE-WINDOW_STRIDE, verbose=False)
        epochs_data = epochs.get_data(copy=False)
    except:
        return None, 0

    if len(epochs_data) == 0: return None, 0
    
    # 3. 提取特征 (n_epochs, n_features)
    feats = extract_features_from_epochs(epochs_data)
    
    # 4. 批量预测
    preds = model.predict(feats) # 返回每个切片的预测类别ID
    
    # 5. 投票机制 (Majority Vote)
    # 统计出现次数最多的类别
    counts = np.bincount(preds.astype(int))
    final_pred_id = np.argmax(counts)
    
    # 计算置信度 (得票率)
    confidence = counts[final_pred_id] / len(preds) * 100
    
    return final_pred_id, confidence

# ==========================================
# 3. 预测主程序
# ==========================================

def main():
    print(">>> [增强版预测] 系统启动...")
    
    if not os.path.exists(MODEL_FILE):
        print(f"错误：找不到模型文件 {MODEL_FILE}")
        return
    
    print(f"Loading model: {MODEL_FILE}...")
    clf = joblib.load(MODEL_FILE)
    
    if not os.path.exists(TEST_DATA_PATH):
        print(f"错误：找不到测试文件夹 {TEST_DATA_PATH}")
        return
        
    files = [f for f in os.listdir(TEST_DATA_PATH) if f.endswith('.txt')]
    if not files:
        print("测试文件夹是空的。")
        return

    print(f"\n{'文件名':<20} | {'预测结果':<15} | {'置信度':<8} | {'真实标签':<15} | {'结果'}")
    print("-" * 85)
    
    y_true = []
    y_pred = []
    
    for f in files:
        path = os.path.join(TEST_DATA_PATH, f)
        
        # 调用投票预测函数
        pred_id, conf = process_file_and_predict(path, clf)
        
        if pred_id is None: continue
        
        pred_name = TARGET_EMOTIONS.get(pred_id, "未知")
        
        # 解析真实标签
        try:
            true_id = int(float(f.split('_')[1].replace('.txt', '')))
            true_name = TARGET_EMOTIONS.get(true_id, "其他")
            
            if true_id in TARGET_EMOTIONS:
                y_true.append(true_id)
                y_pred.append(pred_id)
                is_correct = "✅" if pred_id == true_id else "❌"
            else:
                is_correct = "⚠️" 
        except:
            true_name = "无法解析"
            is_correct = "❓"

        print(f"{f:<20} | {pred_name:<15} | {conf:.1f}%   | {true_name:<15} | {is_correct}")

    if len(y_true) > 0:
        acc = accuracy_score(y_true, y_pred)
        print("-" * 85)
        print(f"\n=== 最终测试报告 ===")
        print(f"样本总数: {len(y_true)}")
        print(f"预测正确: {np.sum(np.array(y_true) == np.array(y_pred))}")
        print(f"整体准确率: {acc * 100:.2f}%")
        
        cm = confusion_matrix(y_true, y_pred, labels=list(TARGET_EMOTIONS.keys()))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[TARGET_EMOTIONS[k] for k in TARGET_EMOTIONS])
        disp.plot(cmap='Blues', xticks_rotation=45)
        plt.title(f"Confusion Matrix (Acc: {acc*100:.1f}%)")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()