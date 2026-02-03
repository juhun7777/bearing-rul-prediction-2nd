import os
import numpy as np
import pandas as pd
import scipy.io
import random
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ────────────────────────────────────────────────────────────
# 0. CWRU 데이터 설정
cwru_data_dir = r"C:\석사 논문\데이터 코드\data\CWRU\raw"

cwru_files = {
    0: "Time_Normal_1_098.mat",      # Normal
    1: "B007_1_123.mat",             # Ball 7mils
    2: "B014_1_190.mat",             # Ball 14mils  
    3: "B021_1_227.mat",             # Ball 21mils
    4: "IR007_1_110.mat",            # IR 7mils
    5: "IR014_1_175.mat",            # IR 14mils
    6: "IR021_1_214.mat",            # IR 21mils
    7: "OR007_6_1_136.mat",          # OR 7mils
    8: "OR014_6_1_202.mat",          # OR 14mils
    9: "OR021_6_1_239.mat",          # OR 21mils
}

# 하이퍼파라미터
SEQ_LEN = 1024          # CWRU용 윈도우 크기
BATCH = 64
EPOCHS = 30
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 시간적 분할 사용
SPLIT_METHOD = 2

print(f"사용 장치: {DEVICE}")
print(f"분할 방법: 시간적 분할 (10-class 전체)")

# ────────────────────────────────────────────────────────────
# 1. 10-class 계층적 라벨 생성

def get_hierarchical_labels_cwru_10class(class_idx):
    """CWRU 10-class 계층적 라벨 생성"""
    if class_idx == 0:  # Normal
        return 0, -1, -1, -1, 0  # main=0, sub1/2/3=-1, full_class=0
    elif 1 <= class_idx <= 3:  # Ball (7, 14, 21 mils)
        full_class = class_idx  # 1, 2, 3
        return 1, class_idx-1, -1, -1, full_class  # main=1, sub1=0,1,2
    elif 4 <= class_idx <= 6:  # IR (7, 14, 21 mils)
        full_class = class_idx  # 4, 5, 6
        return 2, -1, class_idx-4, -1, full_class  # main=2, sub2=0,1,2
    elif 7 <= class_idx <= 9:  # OR (7, 14, 21 mils)
        full_class = class_idx  # 7, 8, 9
        return 3, -1, -1, class_idx-7, full_class  # main=3, sub3=0,1,2

def extract_windows_no_overlap(signal, window_size=1024):
    """오버랩 없이 윈도우 추출"""
    windows = []
    for i in range(0, len(signal) - window_size + 1, window_size):
        window = signal[i:i + window_size]
        windows.append(window)
    return windows

def temporal_split_10class(data_dir, window_size=1024, train_ratio=0.8):
    """시간적 분할로 10-class 데이터 생성"""
    
    print("=== 10-Class 시간적 분할 ===")
    
    train_X, train_labels = [], []
    test_X, test_labels = [], []
    
    for class_idx, filename in cwru_files.items():
        file_path = os.path.join(data_dir, filename)
        
        if not os.path.exists(file_path):
            continue
            
        print(f"로딩: {filename}")
        
        mat_data = scipy.io.loadmat(file_path)
        de_data = None
        for key in mat_data.keys():
            if 'DE_time' in key and not key.startswith('__'):
                de_data = mat_data[key].flatten()
                break
        
        if de_data is None:
            continue
        
        # 시간적 분할 (앞쪽 80% train, 뒤쪽 20% test)
        split_point = int(len(de_data) * train_ratio)
        train_data = de_data[:split_point]
        test_data = de_data[split_point:]
        
        # Train/Test 윈도우 추출 (오버랩 없음)
        train_windows = extract_windows_no_overlap(train_data, window_size)
        test_windows = extract_windows_no_overlap(test_data, window_size)
        
        print(f"  Train 윈도우: {len(train_windows)}개")
        print(f"  Test 윈도우: {len(test_windows)}개")
        
        # 10-class 라벨 생성
        main, s1, s2, s3, full_class = get_hierarchical_labels_cwru_10class(class_idx)
        
        # Train 데이터 추가
        for window in train_windows:
            train_X.append(window.reshape(1, -1))
            train_labels.append((main, s1, s2, s3, full_class))
        
        # Test 데이터 추가
        for window in test_windows:
            test_X.append(window.reshape(1, -1))
            test_labels.append((main, s1, s2, s3, full_class))
    
    return np.array(train_X), train_labels, np.array(test_X), test_labels

def create_balanced_loaders_10class(train_X, train_labels, test_X, test_labels, batch_size=BATCH):
    """10-class용 균형잡힌 데이터로더 생성"""
    
    # 라벨 분리
    train_ym = [label[0] for label in train_labels]
    train_ys1 = [label[1] for label in train_labels]
    train_ys2 = [label[2] for label in train_labels]
    train_ys3 = [label[3] for label in train_labels]
    train_y_full = [label[4] for label in train_labels]  # 10-class 전체 라벨
    
    test_ym = [label[0] for label in test_labels]
    test_ys1 = [label[1] for label in test_labels]
    test_ys2 = [label[2] for label in test_labels]
    test_ys3 = [label[3] for label in test_labels]
    test_y_full = [label[4] for label in test_labels]  # 10-class 전체 라벨
    
    print(f"\nTrain 데이터: {train_X.shape}")
    print(f"Test 데이터: {test_X.shape}")
    
    # 10-class 분포 확인
    print("Train 10-class 분포:")
    for i in range(10):
        count = train_y_full.count(i)
        print(f"  클래스 {i}: {count}개")
    
    print("Test 10-class 분포:")
    for i in range(10):
        count = test_y_full.count(i)
        print(f"  클래스 {i}: {count}개")
    
    # 클래스 균형 맞추기 (가장 적은 클래스에 맞춤)
    min_train_samples = min([train_y_full.count(i) for i in range(10) if train_y_full.count(i) > 0])
    min_test_samples = min([test_y_full.count(i) for i in range(10) if test_y_full.count(i) > 0])
    
    print(f"\n균형 맞추기 - Train: {min_train_samples}개/클래스, Test: {min_test_samples}개/클래스")
    
    # Train 데이터 균형 맞추기
    balanced_train_X, balanced_train_ym = [], []
    balanced_train_ys1, balanced_train_ys2, balanced_train_ys3, balanced_train_y_full = [], [], [], []
    
    for class_i in range(10):
        class_indices = [i for i, y in enumerate(train_y_full) if y == class_i]
        if len(class_indices) > 0:
            selected = random.sample(class_indices, min(len(class_indices), min_train_samples))
            for idx in selected:
                balanced_train_X.append(train_X[idx])
                balanced_train_ym.append(train_ym[idx])
                balanced_train_ys1.append(train_ys1[idx])
                balanced_train_ys2.append(train_ys2[idx])
                balanced_train_ys3.append(train_ys3[idx])
                balanced_train_y_full.append(train_y_full[idx])
    
    # Test 데이터 균형 맞추기
    balanced_test_X, balanced_test_ym = [], []
    balanced_test_ys1, balanced_test_ys2, balanced_test_ys3, balanced_test_y_full = [], [], [], []
    
    for class_i in range(10):
        class_indices = [i for i, y in enumerate(test_y_full) if y == class_i]
        if len(class_indices) > 0:
            selected = random.sample(class_indices, min(len(class_indices), min_test_samples))
            for idx in selected:
                balanced_test_X.append(test_X[idx])
                balanced_test_ym.append(test_ym[idx])
                balanced_test_ys1.append(test_ys1[idx])
                balanced_test_ys2.append(test_ys2[idx])
                balanced_test_ys3.append(test_ys3[idx])
                balanced_test_y_full.append(test_y_full[idx])
    
    # NumPy 배열로 변환
    balanced_train_X = np.array(balanced_train_X)
    balanced_test_X = np.array(balanced_test_X)
    
    print(f"\n균형 후 - Train: {balanced_train_X.shape}, Test: {balanced_test_X.shape}")
    
    # DataLoader 생성
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(balanced_train_X, dtype=torch.float32),
            torch.tensor(balanced_train_ym, dtype=torch.long),
            torch.tensor(balanced_train_ys1, dtype=torch.long),
            torch.tensor(balanced_train_ys2, dtype=torch.long),
            torch.tensor(balanced_train_ys3, dtype=torch.long),
            torch.tensor(balanced_train_y_full, dtype=torch.long),  # 10-class 라벨 추가
        ),
        batch_size=batch_size, shuffle=True
    )
    
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(balanced_test_X, dtype=torch.float32),
            torch.tensor(balanced_test_ym, dtype=torch.long),
            torch.tensor(balanced_test_ys1, dtype=torch.long),
            torch.tensor(balanced_test_ys2, dtype=torch.long),
            torch.tensor(balanced_test_ys3, dtype=torch.long),
            torch.tensor(balanced_test_y_full, dtype=torch.long),  # 10-class 라벨 추가
        ),
        batch_size=batch_size, shuffle=False
    )
    
    return train_loader, test_loader

# ────────────────────────────────────────────────────────────
# 2. 모델 정의 (10-class 출력 추가)

class EnhancedRawCycleMTL_CWRU_10Class(nn.Module):
    def __init__(self, seq_len, num_main=4, num_sub=3,
                 out_main=[4,4,4,4], out_sub=[3,3,3], out_full=10):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        hidden_dim = 256
        self.fc = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.main_focus = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU()
        )
        
        self.main_pen = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_main)])
        self.main_clf = nn.ModuleList([nn.Linear(hidden_dim, od) for od in out_main])
        
        self.mha_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
            for _ in range(num_main)
        ])
        
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.query_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_sub)])
        self.sub_clf = nn.ModuleList([nn.Linear(2*hidden_dim, od) for od in out_sub])
        
        # 10-class 전체 분류기 추가
        self.full_clf = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, out_full)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        h_cnn = self.cnn(x)
        h_pooled = self.global_pool(h_cnn).squeeze(-1)
        h = self.fc(h_pooled)
        
        main_enhanced = self.main_focus(h)
        
        sub_queries = [q_proj(h) for q_proj in self.query_proj]
        sub_knowledge = sum(sub_queries) / len(sub_queries)
        
        m, lm = [], []
        for i, (pen, clf, mha) in enumerate(zip(self.main_pen, self.main_clf, self.mha_layers)):
            enhanced_h = main_enhanced + 0.3 * sub_knowledge
            enhanced_h = enhanced_h.unsqueeze(1)
            attn_out, _ = mha(enhanced_h, enhanced_h, enhanced_h)
            attn_out = attn_out.squeeze(1)
            
            mi = F.relu(pen(attn_out))
            m.append(mi)
            lm.append(clf(mi))
        
        keys = [self.key_proj(mi) for mi in m]
        ls = []
        for q_proj, clf in zip(self.query_proj, self.sub_clf):
            q = q_proj(h)
            scores = torch.stack([(q*k).sum(-1) for k in keys], dim=1)
            alpha = F.softmax(scores, dim=1)
            c = sum(alpha[:,i:i+1]*m[i] for i in range(len(m)))
            inp = torch.cat([h, c], dim=1)
            ls.append(clf(inp))
        
        # 10-class 전체 분류
        lf = self.full_clf(h)
        
        return lm, ls, lf

# ────────────────────────────────────────────────────────────
# 3. 손실 및 평가 함수

def loss_fn_cwru_10class(lm, ls, lf, ym, ys1, ys2, ys3, yf, main_weight=3.0, full_weight=2.0):
    loss = 0
    
    # 메인 태스크 손실
    for out in lm:
        loss += main_weight * F.cross_entropy(out, ym)
    
    # 서브 태스크 손실
    for out, ys in zip(ls, (ys1, ys2, ys3)):
        mask = (ys >= 0)
        if mask.any():
            loss += F.cross_entropy(out[mask], ys[mask])
    
    # 10-class 전체 손실
    loss += full_weight * F.cross_entropy(lf, yf)
    
    return loss

def evaluate_all_tasks(model, test_loader, device=DEVICE):
    """메인, 서브, 10-class 전체 성능 평가"""
    model.eval()
    
    all_main_preds, all_main_trues = [], []
    all_full_preds, all_full_trues = [], []
    
    with torch.no_grad():
        for x, ym, ys1, ys2, ys3, yf in test_loader:
            x = x.to(device)
            ym_true = ym.cpu().numpy()
            yf_true = yf.cpu().numpy()
            
            lm, ls, lf = model(x)
            
            # 메인 태스크 예측
            main_preds = lm[0].argmax(dim=1).cpu().numpy()
            all_main_preds.extend(main_preds)
            all_main_trues.extend(ym_true)
            
            # 10-class 전체 예측
            full_preds = lf.argmax(dim=1).cpu().numpy()
            all_full_preds.extend(full_preds)
            all_full_trues.extend(yf_true)
    
    return (np.array(all_main_preds), np.array(all_main_trues), 
            np.array(all_full_preds), np.array(all_full_trues))

# ────────────────────────────────────────────────────────────
# 4. 메인 실행 코드

if __name__ == "__main__":
    # 시드 설정
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("=== CWRU 10-Class 계층적 학습 시작 ===")
    
    # 1. 데이터 분할
    train_X, train_labels, test_X, test_labels = temporal_split_10class(
        cwru_data_dir, SEQ_LEN, train_ratio=0.8
    )
    
    # 2. 균형잡힌 데이터로더 생성
    train_loader, test_loader = create_balanced_loaders_10class(
        train_X, train_labels, test_X, test_labels
    )
    
    # 3. 모델 초기화
    model = EnhancedRawCycleMTL_CWRU_10Class(seq_len=SEQ_LEN).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3)
    
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. 학습
    print("\n=== 10-Class 계층적 모델 학습 시작 ===")
    best_main_acc = 0
    best_full_acc = 0
    
    for epoch in range(1, EPOCHS+1):
        model.train()
        train_loss = 0
        for x, ym, ys1, ys2, ys3, yf in train_loader:
            x, ym, ys1, ys2, ys3, yf = [t.to(DEVICE) for t in (x,ym,ys1,ys2,ys3,yf)]
            opt.zero_grad()
            lm, ls, lf = model(x)
            loss = loss_fn_cwru_10class(lm, ls, lf, ym, ys1, ys2, ys3, yf, 
                                      main_weight=3.0, full_weight=2.0)
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # 평가
        main_preds, main_trues, full_preds, full_trues = evaluate_all_tasks(model, test_loader)
        main_acc = accuracy_score(main_trues, main_preds)
        full_acc = accuracy_score(full_trues, full_preds)
        
        print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | Main: {main_acc:.4f} | 10-Class: {full_acc:.4f}")
        
        if main_acc > best_main_acc:
            best_main_acc = main_acc
        if full_acc > best_full_acc:
            best_full_acc = full_acc
        
        scheduler.step(full_acc)  # 10-class 성능 기준으로 스케줄링
    
    # 5. 최종 평가
    print(f"\n=== 최종 10-Class 계층적 결과 ===")
    
    main_preds, main_trues, full_preds, full_trues = evaluate_all_tasks(model, test_loader)
    final_main_acc = accuracy_score(main_trues, main_preds)
    final_full_acc = accuracy_score(full_trues, full_preds)
    
    print(f"최고 메인 정확도 (4-class): {best_main_acc:.4f}")
    print(f"최고 10-class 정확도: {best_full_acc:.4f}")
    print(f"최종 메인 정확도 (4-class): {final_main_acc:.4f}")
    print(f"최종 10-class 정확도: {final_full_acc:.4f}")
    
    # 혼동행렬과 성능 분석
    cm_main = confusion_matrix(main_trues, main_preds)
    cm_full = confusion_matrix(full_trues, full_preds)
    
    print("\n=== 메인 태스크 (4-class) 혼동행렬 ===")
    print("0:Normal, 1:Ball, 2:IR, 3:OR")
    print(cm_main)
    
    print("\n=== 10-Class 전체 혼동행렬 ===")
    print("0:Normal, 1-3:Ball(7,14,21), 4-6:IR(7,14,21), 7-9:OR(7,14,21)")
    print(cm_full)
    
    # 클래스별 성능 분석
    class_names_4 = ['Normal', 'Ball', 'Inner-race', 'Outer-race']
    class_names_10 = ['Normal', 'Ball_7', 'Ball_14', 'Ball_21', 
                      'IR_7', 'IR_14', 'IR_21', 'OR_7', 'OR_14', 'OR_21']
    
    report_main = classification_report(main_trues, main_preds, target_names=class_names_4)
    report_full = classification_report(full_trues, full_preds, target_names=class_names_10)
    
    print("\n=== 메인 태스크 (4-class) 성능 ===")
    print(report_main)
    
    print("\n=== 10-Class 전체 성능 ===")
    print(report_full)
    
    # 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 4-class 혼동행렬
    sns.heatmap(cm_main, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=class_names_4, yticklabels=class_names_4)
    ax1.set_title(f'메인 태스크 (4-class)\n정확도: {final_main_acc:.4f}')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    
    # 10-class 혼동행렬
    sns.heatmap(cm_full, annot=True, fmt='d', cmap='Reds', ax=ax2,
                xticklabels=class_names_10, yticklabels=class_names_10)
    ax2.set_title(f'10-Class 전체\n정확도: {final_full_acc:.4f}')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== CWRU 10-Class 학습 완료 ===")
    print(f"메인 태스크 (4-class): {final_main_acc:.4f}")
    print(f"10-Class 전체: {final_full_acc:.4f}")
    
    # 성능 평가 메시지
    if final_full_acc > 0.90:
        print("🎉 10-class에서 90% 이상! 정말 우수한 성능입니다!")
    elif final_full_acc > 0.85:
        print("✅ 10-class에서 85% 이상! 좋은 성능입니다!")
    elif final_full_acc > 0.80:
        print("👍 10-class에서 80% 이상! 합리적인 성능입니다!")
    else:
        print("⚠️  10-class는 4-class보다 훨씬 어려운 문제입니다. 추가 튜닝을 고려해보세요.")