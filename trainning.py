import os
import numpy as np
import mne
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ==========================================
# 1. 配置区域
# ==========================================
TRAIN_DATA_PATH = 'eeg_raw'

MODEL_FILE = 'bci_5class_model_boost.pkl' 

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

ASYMMETRY_PAIRS = [(0, 13), (1, 12), (2, 11), (3, 10), (4, 9), (5, 8), (6, 7)]
# 滑动窗口配置

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
    from scipy.signal import welch
    n_epochs, n_channels, n_times = epochs_data.shape
    # 【修复策略1】降低最高频率到 40Hz，减少肌肉噪音干扰
    freqs, psds = welch(epochs_data, fs=SFREQ, nperseg=SFREQ, axis=-1)


    # 频段定义
    freq_bands = {'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 40)}
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

# ==========================================
# 3. 训练主程序
# ==========================================

def main():
    print(f">>> [均衡版训练] 启动...")
    print("    策略: 启用 class_weight='balanced' (修正Fear偏见)")
    print("    策略: 限制 Gamma < 40Hz (抗噪)")
    if not os.path.exists(TRAIN_DATA_PATH):

        print(f"错误：找不到 {TRAIN_DATA_PATH}")

        return

    X_all, y_all = [], []
    groups = []
    files = [f for f in os.listdir(TRAIN_DATA_PATH) if f.endswith('.txt')]

    count_dict = {name: 0 for name in TARGET_EMOTIONS.values()}

    # 1. 加载数据
    for i, f in enumerate(files):

        try:

            eid = int(float(f.split('_')[1].replace('.txt', '')))

            if eid in TARGET_EMOTIONS:

                path = os.path.join(TRAIN_DATA_PATH, f)

                raw = load_txt_to_mne(path)

                if raw is None: continue

                # 预处理
                raw.set_eeg_reference('average', projection=False, verbose=False)
                # 【抗噪】滤波到 40Hz 即可
                raw.filter(l_freq=1.0, h_freq=40.0, verbose=False)
                epochs = mne.make_fixed_length_epochs(raw, duration=WINDOW_SIZE, overlap=WINDOW_SIZE-WINDOW_STRIDE, verbose=False)

                epochs_data = epochs.get_data(copy=False)

                if len(epochs_data) > 0:

                    feats = extract_features_from_epochs(epochs_data)

                    X_all.append(feats)

                    y_all.extend([eid] * len(epochs_data))

                    groups.extend([f] * len(epochs_data))

                    count_dict[TARGET_EMOTIONS[eid]] += len(epochs_data)

        except: continue
        if i % 20 == 0: print(f"    扫描进度: {i}/{len(files)}", end='\r')

    if not X_all: return
    X = np.vstack(X_all)
    y = np.array(y_all)
    groups = np.array(groups)
    print(f"\n\n>>> 样本分布:")

    for name, count in count_dict.items():

        print(f"    - {name}: {count}")

    # 2. 训练模型

    print("\n>>> 正在训练均衡模型...")
    # 【核心修复】class_weight='balanced'

    # 这会告诉 SVM：“Fear 样本太多了，不值钱；Joy 样本少，很珍贵”，强制它平衡注意力。

    clf = make_pipeline(

        StandardScaler(),

        SVC(kernel='rbf', C=5.0, probability=True, class_weight='balanced')

    )

    # 3. 内部评估 (看混淆矩阵)
    gkf = StratifiedGroupKFold(n_splits=5)

    y_pred_cv = cross_val_predict(clf, X, y, groups=groups, cv=gkf)
    print("\n>>> 训练集内部评估 (修正后):")
    print(classification_report(y, y_pred_cv, target_names=[TARGET_EMOTIONS[k] for k in sorted(TARGET_EMOTIONS.keys())]))

    # 4. 全量训练并保存
    clf.fit(X, y)
    joblib.dump(clf, MODEL_FILE)
    print(f"✅ 模型已修复并保存: {MODEL_FILE}")
    print("请使用之前的预测代码重新测试！")

if __name__ == "__main__":

    main()