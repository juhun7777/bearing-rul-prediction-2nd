import os
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 폰트 설정 (영어 표기용)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ────────────────────────────────────────────────────────────
# 0. 하이퍼파라미터 및 경로 설정 - 증강된 데이터 사용
augmented_data_dir = r"C:/석사 논문/데이터 코드/data/augmented_data/"

train_paths = {
    0: os.path.join(augmented_data_dir, "normal_all3_train800.csv"),
    1: os.path.join(augmented_data_dir, "300_train800.csv"),
    2: os.path.join(augmented_data_dir, "260_train800.csv"),
    4: os.path.join(augmented_data_dir, "belt1_train800.csv"),
    5: os.path.join(augmented_data_dir, "belt2_train800.csv"),
    6: os.path.join(augmented_data_dir, "belt3_train800.csv"),
}

test_paths = {
    0: os.path.join(augmented_data_dir, "normal_all3_test200.csv"),
    1: os.path.join(augmented_data_dir, "300_test200.csv"),
    2: os.path.join(augmented_data_dir, "260_test200.csv"),
    4: os.path.join(augmented_data_dir, "belt1_test200.csv"),
    5: os.path.join(augmented_data_dir, "belt2_test200.csv"),
    6: os.path.join(augmented_data_dir, "belt3_test200.csv"),
}

SEQ_LEN  = 7800         # cycle 당 시퀀스 길이
BATCH    = 64
EPOCHS   = 30
LR       = 1e-3
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

print(f"사용 장치: {DEVICE}")

# ────────────────────────────────────────────────────────────
# 1. 증강된 데이터 로드 및 전처리 - 6개 클래스 사용
def load_augmented_cycles(train_paths, test_paths, seq_len=SEQ_LEN):
    """증강된 train/test 데이터를 별도로 로드"""
    
    def process_paths(paths, dataset_type):
        X_list, y_main, y_s1, y_s2, y_s3 = [], [], [], [], []
        
        for lbl, path in paths.items():
            print(f"로딩 중: {dataset_type} - {os.path.basename(path)}")
            
            if not os.path.exists(path):
                print(f"경고: 파일이 존재하지 않습니다 - {path}")
                continue
                
            df = pd.read_csv(path)
            print(f"  데이터 크기: {len(df)} 행, 사이클 수: {df['cycle_id'].nunique()}")
            
            for cid, grp in df.groupby("cycle_id"):
                arr = grp[["x","y"]].to_numpy(float)
                # 패딩/트리밍
                if arr.shape[0] < seq_len:
                    pad = np.zeros((seq_len - arr.shape[0], 2))
                    arr = np.vstack([arr, pad])
                else:
                    arr = arr[:seq_len]
                X_list.append(arr.T)  # (2, seq_len)
                
                # 라벨링 - Tension은 2개 클래스 (300, 260)
                main = 0 if lbl==0 else (1 if lbl in (1,2) else 2)
                if main == 0:
                    s1,s2,s3 = -1,-1,-1
                elif main == 1:
                    s1 = {1:0,2:1}[lbl]; s2,s3 = -1,-1  # Tension은 2개 클래스 (300, 260)
                else:
                    s2 = {4:0,5:1,6:2}[lbl]; s1,s3 = -1,-1
                y_main.append(main)
                y_s1.append(s1)
                y_s2.append(s2)
                y_s3.append(s3)
        
        if len(X_list) == 0:
            return None, None, None, None, None
            
        X = np.stack(X_list)  # (N,2,seq_len)
        return X, np.array(y_main), np.array(y_s1), np.array(y_s2), np.array(y_s3)
    
    # Train 데이터 로드
    print("=== Train 데이터 로딩 ===")
    X_tr, ym_tr, ys1_tr, ys2_tr, ys3_tr = process_paths(train_paths, "Train")
    
    # Test 데이터 로드
    print("\n=== Test 데이터 로딩 ===")
    X_te, ym_te, ys1_te, ys2_te, ys3_te = process_paths(test_paths, "Test")
    
    return X_tr, X_te, ym_tr, ym_te, ys1_tr, ys1_te, ys2_tr, ys2_te, ys3_tr, ys3_te

# 증강된 데이터 로드
X_tr, X_te, ym_tr, ym_te, ys1_tr, ys1_te, ys2_tr, ys2_te, ys3_tr, ys3_te = load_augmented_cycles(train_paths, test_paths)

# 데이터 분포 확인
print("\n=== 데이터 분포 ===")
print(f"Train 샘플 수: {len(ym_tr)}")
print(f"Test 샘플 수: {len(ym_te)}")
print(f"Train 메인 클래스 분포: Normal={sum(ym_tr==0)}, Tension={sum(ym_tr==1)}, Wear={sum(ym_tr==2)}")
print(f"Test 메인 클래스 분포: Normal={sum(ym_te==0)}, Tension={sum(ym_te==1)}, Wear={sum(ym_te==2)}")
print(f"Train Tension 서브클래스 분포: 300={sum(ys1_tr==0)}, 260={sum(ys1_tr==1)}")
print(f"Train Wear 서브클래스 분포: belt1={sum(ys2_tr==0)}, belt2={sum(ys2_tr==1)}, belt3={sum(ys2_tr==2)}")

def make_loader(X, ym, ys1, ys2, ys3, batch=BATCH, shuffle=True):
    return DataLoader(
        TensorDataset(
            torch.tensor(X,  dtype=torch.float32),
            torch.tensor(ym, dtype=torch.long),
            torch.tensor(ys1,dtype=torch.long),
            torch.tensor(ys2,dtype=torch.long),
            torch.tensor(ys3,dtype=torch.long),
        ),
        batch_size=batch, shuffle=shuffle
    )

train_loader = make_loader(X_tr, ym_tr, ys1_tr, ys2_tr, ys3_tr)
test_loader  = make_loader(X_te, ym_te, ys1_te, ys2_te, ys3_te, shuffle=False)

print(f"Train 배치 수: {len(train_loader)}")
print(f"Test 배치 수: {len(test_loader)}")

# ────────────────────────────────────────────────────────────
# 2. Hessian 기반 적응적 가중치 옵티마이저
class AdaptiveWeightOptimizer:
    """가중치 파라미터에 대한 적응적 2차 최적화 (Hessian 근사)"""
    def __init__(self, weight_param, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, hessian_reg=1e-4):
        self.weight_param = weight_param
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.hessian_reg = hessian_reg
        
        # Momentum 저장용
        self.m = torch.zeros_like(weight_param)
        self.v = torch.zeros_like(weight_param)
        self.t = 0
        
        # Hessian 근사를 위한 gradient history
        self.grad_history = []
        self.max_history = 10
    
    def step(self):
        if self.weight_param.grad is None:
            return
            
        self.t += 1
        grad = self.weight_param.grad.data.clone()
        
        # Gradient history 업데이트
        self.grad_history.append(grad.clone())
        if len(self.grad_history) > self.max_history:
            self.grad_history.pop(0)
        
        # Adam 모멘텀 업데이트
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad.pow(2)
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Hessian 대각선 근사 (gradient 변화율 기반)
        hessian_diag = v_hat.sqrt() + self.eps
        
        if len(self.grad_history) >= 2:
            # Secant method를 사용한 Hessian 근사 개선
            grad_diff = self.grad_history[-1] - self.grad_history[-2]
            param_diff = self.lr * m_hat  # 이전 스텝에서의 파라미터 변화량 근사
            
            # Hessian 대각선 근사 개선 (BFGS 스타일)
            if param_diff.abs() > 1e-10:
                secant_hessian = grad_diff.abs() / (param_diff.abs() + self.eps)
                # 기존 추정치와 결합
                hessian_diag = 0.7 * hessian_diag + 0.3 * (secant_hessian + self.hessian_reg)
        
        # 적응적 학습률 (Hessian 역수 사용)
        adapted_lr = self.lr / hessian_diag
        
        # Natural gradient 방향으로 업데이트
        natural_grad = m_hat / hessian_diag
        
        # 파라미터 업데이트 (클리핑으로 안정성 확보)
        update = torch.clamp(adapted_lr * natural_grad, -0.1, 0.1)
        self.weight_param.data -= update
        
        # 가중치를 합리적 범위로 제한 (시그모이드 적용 전 raw 값)
        self.weight_param.data = torch.clamp(self.weight_param.data, -5.0, 5.0)
    
    def zero_grad(self):
        if self.weight_param.grad is not None:
            self.weight_param.grad.zero_()
    
    def get_hessian_info(self):
        """현재 Hessian 근사 정보 반환"""
        if hasattr(self, 'v') and self.t > 0:
            v_hat = self.v / (1 - self.beta2 ** self.t)
            return v_hat.sqrt().item()
        return 0.0

# ────────────────────────────────────────────────────────────
# 3. Hessian 기반 학습 가능한 가중치를 가진 MTL 모델
class HessianWeightMTL(nn.Module):
    def __init__(self, seq_len, num_main=3, num_sub=3,
                 out_main=[3,3,3], out_sub=[2,3,3]):
        super().__init__()
        
        # CNN 백본
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(32),
            nn.MaxPool1d(2),  # seq_len/2
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(64),
            nn.MaxPool1d(2),  # seq_len/4
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.MaxPool1d(2),  # seq_len/8
        )
        
        # 전역 풀링
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 특징 처리 레이어
        hidden_dim = 256
        self.fc = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 메인 태스크 강화 모듈
        self.main_focus = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU()
        )
        
        # === Hessian 기반 학습 가능한 지식 전달 가중치 ===
        # 초기값을 logit(0.3) ≈ -0.847로 설정하여 시그모이드 적용 시 0.3이 되도록 함
        initial_weight = torch.log(torch.tensor(0.3) / (1 - 0.3))  # logit 변환
        self.knowledge_weight = nn.Parameter(initial_weight)
        
        # Main tasks
        self.main_pen = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_main)])
        self.main_clf = nn.ModuleList([nn.Linear(hidden_dim, od) for od in out_main])
        
        # 멀티헤드 어텐션
        self.mha_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
            for _ in range(num_main)
        ])
        
        # 서브태스크 관련
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.query_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_sub)])
        self.sub_clf = nn.ModuleList([nn.Linear(2*hidden_dim, od) for od in out_sub])
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        # CNN 특징 추출
        h_cnn = self.cnn(x)  # [batch, 128, seq_len/8]
        h_pooled = self.global_pool(h_cnn).squeeze(-1)  # [batch, 128]
        h = self.fc(h_pooled)  # [batch, 256]
        
        # 메인 태스크 강화
        main_enhanced = self.main_focus(h)  # [batch, 256]
        
        # 서브태스크 지식 추출
        sub_queries = [q_proj(h) for q_proj in self.query_proj]
        sub_knowledge = sum(sub_queries) / len(sub_queries)  # 평균 서브태스크 지식
        
        # === Hessian 기반 학습된 가중치 적용 ===
        # 시그모이드를 통해 0~1 범위로 변환
        learned_weight = torch.sigmoid(self.knowledge_weight)
        
        # 메인 태스크 처리 (학습된 가중치로 서브태스크 지식 결합)
        m, lm = [], []
        for i, (pen, clf, mha) in enumerate(zip(self.main_pen, self.main_clf, self.mha_layers)):
            # 학습된 가중치로 지식 전달
            enhanced_h = main_enhanced + learned_weight * sub_knowledge
            
            # 멀티헤드 어텐션 적용
            enhanced_h = enhanced_h.unsqueeze(1)  # [batch, 1, 256]
            attn_out, _ = mha(enhanced_h, enhanced_h, enhanced_h)
            attn_out = attn_out.squeeze(1)  # [batch, 256]
            
            # 메인 태스크 분류
            mi = F.relu(pen(attn_out))
            m.append(mi)
            lm.append(clf(mi))
        
        # 서브태스크 처리
        keys = [self.key_proj(mi) for mi in m]
        ls = []
        for q_proj, clf in zip(self.query_proj, self.sub_clf):
            q = q_proj(h)
            scores = torch.stack([(q*k).sum(-1) for k in keys], dim=1)
            alpha = F.softmax(scores, dim=1)
            c = sum(alpha[:,i:i+1]*m[i] for i in range(len(m)))
            inp = torch.cat([h, c], dim=1)
            ls.append(clf(inp))
        
        return lm, ls
    
    def get_current_weight(self):
        """현재 학습된 가중치 값 반환"""
        return torch.sigmoid(self.knowledge_weight).item()

# ────────────────────────────────────────────────────────────
# 4. 모델 초기화 및 옵티마이저 설정
model = HessianWeightMTL(seq_len=SEQ_LEN, out_sub=[2,3,3]).to(DEVICE)

# 일반 파라미터와 가중치 파라미터 분리
weight_params = []
other_params = []

for name, param in model.named_parameters():
    if 'knowledge_weight' in name:
        weight_params.append(param)
        print(f"가중치 파라미터 발견: {name}, 초기값: {torch.sigmoid(param).item():.4f}")
    else:
        other_params.append(param)

# 서로 다른 옵티마이저 사용
main_optimizer = torch.optim.Adam(other_params, lr=LR)

# Hessian 기반 가중치 옵티마이저 (학습률을 높게 설정)
weight_optimizers = [AdaptiveWeightOptimizer(wp, lr=0.1, beta1=0.9, beta2=0.999) 
                     for wp in weight_params]

print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
print(f"학습 가능한 가중치 파라미터 수: {len(weight_params)}")

# ────────────────────────────────────────────────────────────
# 5. 손실 함수
def loss_fn(lm, ls, ym, ys1, ys2, ys3, main_weight=2.0):
    loss = 0
    for out in lm:
        loss += main_weight * F.cross_entropy(out, ym)
    for out, ys in zip(ls, (ys1, ys2, ys3)):
        mask = (ys>=0)
        if mask.any():
            loss += F.cross_entropy(out[mask], ys[mask])
    return loss

# ────────────────────────────────────────────────────────────
# 6. 학습 및 가중치 추적
train_losses, val_accs = [], []
weight_history = []  # 가중치 변화 추적
hessian_history = []  # Hessian 정보 추적

print("\n=== Hessian 기반 가중치 학습 시작 ===")
print(f"초기 지식 전달 가중치: {model.get_current_weight():.4f}")

for epoch in range(1, EPOCHS+1):
    model.train()
    train_loss = 0
    epoch_weights = []
    
    for batch_idx, (x, ym, ys1, ys2, ys3) in enumerate(train_loader):
        x, ym, ys1, ys2, ys3 = [t.to(DEVICE) for t in (x,ym,ys1,ys2,ys3)]
        
        # Forward pass
        lm, ls = model(x)
        loss = loss_fn(lm, ls, ym, ys1, ys2, ys3, main_weight=2.0)
        
        # Backward pass
        main_optimizer.zero_grad()
        for opt in weight_optimizers:
            opt.zero_grad()
        
        loss.backward()
        
        # 파라미터 업데이트
        main_optimizer.step()
        for opt in weight_optimizers:
            opt.step()
        
        train_loss += loss.item() * x.size(0)
        
        # 배치별 가중치 기록
        epoch_weights.append(model.get_current_weight())
    
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)
    
    # 에포크별 가중치 정보 저장
    avg_weight = np.mean(epoch_weights)
    weight_history.append(avg_weight)
    
    # Hessian 정보 저장
    if weight_optimizers:
        hessian_info = weight_optimizers[0].get_hessian_info()
        hessian_history.append(hessian_info)
    
    # 검증
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, ym, *_ in test_loader:
            x = x.to(DEVICE)
            lm, _ = model(x)
            preds.append(lm[0].argmax(1).cpu().numpy())
            trues.append(ym.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    acc = accuracy_score(trues, preds)
    val_accs.append(acc)
    
    # 에포크별 상세 정보 출력
    current_weight = model.get_current_weight()
    raw_weight = model.knowledge_weight.item()
    print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | Val Acc: {acc:.4f} | "
          f"Weight: {current_weight:.4f} (raw: {raw_weight:.4f}) | "
          f"Hessian: {hessian_info:.6f}")

print(f"\n최종 학습된 지식 전달 가중치: {model.get_current_weight():.4f}")
print(f"가중치 변화량: {abs(weight_history[-1] - weight_history[0]):.4f}")

# ────────────────────────────────────────────────────────────
# 7. 결과 시각화
plt.figure(figsize=(20, 12))

# 학습 곡선
plt.subplot(2, 3, 1)
plt.plot(train_losses, 'b-', linewidth=2)
plt.title("Training Loss", fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 2)
plt.plot(val_accs, 'r-', linewidth=2)
plt.title("Validation Accuracy", fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.grid(True, alpha=0.3)

# 가중치 변화 추적
plt.subplot(2, 3, 3)
plt.plot(weight_history, 'g-', linewidth=2, label='Knowledge Weight')
plt.axhline(y=0.3, color='orange', linestyle='--', label='Initial (0.3)')
plt.title("Knowledge Transfer Weight Evolution\n(Hessian-based Optimization)", fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Weight Value", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# Hessian 정보 변화
plt.subplot(2, 3, 4)
plt.plot(hessian_history, 'm-', linewidth=2)
plt.title("Hessian Diagonal Approximation", fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Hessian Value", fontsize=12)
plt.grid(True, alpha=0.3)

# 가중치 분포 (히스토그램)
plt.subplot(2, 3, 5)
plt.hist(weight_history, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(x=0.3, color='orange', linestyle='--', linewidth=2, label='Initial (0.3)')
plt.axvline(x=weight_history[-1], color='red', linestyle='-', linewidth=2, label=f'Final ({weight_history[-1]:.3f})')
plt.title("Weight Value Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Weight Value", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# 학습률 효과 (Loss vs Weight)
plt.subplot(2, 3, 6)
plt.scatter(weight_history, train_losses, c=range(len(weight_history)), cmap='viridis', alpha=0.7)
plt.colorbar(label='Epoch')
plt.title("Loss vs Knowledge Weight", fontsize=14, fontweight='bold')
plt.xlabel("Knowledge Weight", fontsize=12)
plt.ylabel("Training Loss", fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────────────────
# 8. 최종 평가 (기존과 동일)
model.eval()
all_main_preds = []
all_main_trues = []
sub_preds_by_main = {1: [], 2: []}
sub_trues_by_main = {1: [], 2: []}

with torch.no_grad():
    for x, y_main, y_s1, y_s2, y_s3 in test_loader:
        x = x.to(DEVICE)
        logits_main, logits_sub = model(x)
        
        preds_main = logits_main[0].argmax(dim=1).cpu().numpy()
        trues_main = y_main.numpy()
        all_main_preds.append(preds_main)
        all_main_trues.append(trues_main)
        
        preds_sub1 = logits_sub[0].argmax(dim=1).cpu().numpy()
        preds_sub2 = logits_sub[1].argmax(dim=1).cpu().numpy()
        trues_s1 = y_s1.numpy()
        trues_s2 = y_s2.numpy()
        
        for pm, ps1, ts1, ps2, ts2 in zip(preds_main, preds_sub1, trues_s1, preds_sub2, trues_s2):
            if pm == 1 and ts1 >= 0:
                sub_preds_by_main[1].append(ps1)
                sub_trues_by_main[1].append(ts1)
            elif pm == 2 and ts2 >= 0:
                sub_preds_by_main[2].append(ps2)
                sub_trues_by_main[2].append(ts2)

all_main_preds = np.concatenate(all_main_preds)
all_main_trues = np.concatenate(all_main_trues)

main_acc = accuracy_score(all_main_trues, all_main_preds)
print(f"\n=== Hessian 기반 최종 결과 ===")
print(f"Overall Main Accuracy: {main_acc:.4f}")
print(f"최종 학습된 지식 전달 가중치: {model.get_current_weight():.4f}")

# 메인 혼동 행렬
cm_main = confusion_matrix(all_main_trues, all_main_preds, labels=[0,1,2])
print("Main Confusion Matrix:")
print(cm_main)

# 서브태스크 정확도
for main_cls in [1, 2]:
    if len(sub_trues_by_main[main_cls]) > 0:
        sub_acc = accuracy_score(sub_trues_by_main[main_cls], sub_preds_by_main[main_cls])
        print(f"Sub-task Accuracy for main={main_cls}: {sub_acc:.4f}")
        # 서브태스크 혼동 행렬
        if main_cls == 1:  # Tension
            n_classes = 2  # 300, 260
            cm_sub = confusion_matrix(sub_trues_by_main[main_cls], sub_preds_by_main[main_cls], labels=range(n_classes))
            print(f"Tension Sub-task Confusion Matrix:")
            print(cm_sub)
        else:  # Wear
            n_classes = 3  # belt1, belt2, belt3
            cm_sub = confusion_matrix(sub_trues_by_main[main_cls], sub_preds_by_main[main_cls], labels=range(n_classes))
            print(f"Wear Sub-task Confusion Matrix:")
            print(cm_sub)
    else:
        print(f"No samples for sub-task main={main_cls}")

# 6클래스 계층적 예측 생성
hierarchical_preds = []
hierarchical_trues = []

for i in range(len(ym_te)):
    # 실제 클래스
    if ym_te[i] == 0:  # Normal
        true_label = 0
    elif ym_te[i] == 1:  # Tension
        true_label = 1 + ys1_te[i]  # 1(300), 2(260)
    else:  # Wear
        true_label = 3 + ys2_te[i]  # 3(belt1), 4(belt2), 5(belt3)
    
    hierarchical_trues.append(true_label)
    
    # 예측 클래스
    main_pred = all_main_preds[i]
    if main_pred == 0:  # Normal
        pred_label = 0
    elif main_pred == 1:  # Tension
        matching_indices = [j for j, (pred, true) in enumerate(zip(all_main_preds, all_main_trues)) 
                           if pred == 1 and true == 1]
        
        if i in matching_indices:
            idx_in_sub = matching_indices.index(i)
            if idx_in_sub < len(sub_preds_by_main[1]):
                tension_pred = sub_preds_by_main[1][idx_in_sub]
                pred_label = 1 + tension_pred  # 1(300), 2(260)
            else:
                pred_label = 1  # 기본값 300
        else:
            pred_label = 1  # 기본값 300
    else:  # Wear
        matching_indices = [j for j, (pred, true) in enumerate(zip(all_main_preds, all_main_trues)) 
                           if pred == 2 and true == 2]
        
        if i in matching_indices:
            idx_in_sub = matching_indices.index(i)
            if idx_in_sub < len(sub_preds_by_main[2]):
                wear_pred = sub_preds_by_main[2][idx_in_sub]
                pred_label = 3 + wear_pred  # 3(belt1), 4(belt2), 5(belt3)
            else:
                pred_label = 3  # 기본값 belt1
        else:
            pred_label = 3  # 기본값 belt1
    
    hierarchical_preds.append(pred_label)

# 6클래스 계층적 분류 정확도
hierarchical_acc = accuracy_score(hierarchical_trues, hierarchical_preds)
print(f"\n6-class Hierarchical Accuracy: {hierarchical_acc:.4f}")

# 6클래스 혼동 행렬
class_names = ['Normal', 'T_300', 'T_260', 'B_1', 'B_2', 'B_3']
cm_hierarchical = confusion_matrix(hierarchical_trues, hierarchical_preds, labels=range(6))
print("6-class Hierarchical Confusion Matrix:")
print(cm_hierarchical)

# 6클래스 혼동 행렬 시각화
plt.figure(figsize=(12, 10))
im = plt.imshow(cm_hierarchical, cmap='Blues', interpolation='nearest')

# 컬러바 추가
cbar = plt.colorbar(im)
cbar.ax.tick_params(labelsize=12)
cbar.set_label('Sample Count', fontsize=12)

# 축 라벨 설정
plt.xticks(range(6), class_names, rotation=45, fontsize=12, ha='right')
plt.yticks(range(6), class_names, fontsize=12)
plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=14, fontweight='bold')
plt.title(f'6-Class Hierarchical Classification Confusion Matrix\n(Hessian-based Adaptive Weight: {model.get_current_weight():.4f})', 
          fontsize=16, fontweight='bold', pad=20)

# 각 셀에 값 표시
thresh = cm_hierarchical.max() / 2
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, format(cm_hierarchical[i, j], 'd'),
                 ha="center", va="center",
                 fontsize=16,
                 fontweight='bold',
                 color="white" if cm_hierarchical[i, j] > thresh else "black")

# 정확도 및 가중치 정보 추가
plt.figtext(0.02, 0.02, f'Overall Accuracy: {hierarchical_acc:.4f} ({hierarchical_acc*100:.2f}%)\nLearned Weight: {model.get_current_weight():.4f}', 
            fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

plt.tight_layout()
plt.show()

# 클래스별 성능 분석
print("\n=== Class-wise Performance Analysis ===")
for i, class_name in enumerate(class_names):
    tp = cm_hierarchical[i, i]
    fp = cm_hierarchical[:, i].sum() - tp
    fn = cm_hierarchical[i, :].sum() - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"{class_name}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

# 가중치 학습 결과 요약
print(f"\n=== Hessian 기반 가중치 학습 요약 ===")
print(f"초기 가중치: 0.3000")
print(f"최종 가중치: {model.get_current_weight():.4f}")
print(f"가중치 변화량: {abs(model.get_current_weight() - 0.3):.4f}")
print(f"최종 Hessian 근사값: {hessian_history[-1]:.6f}")
print(f"Train 데이터: {len(ym_tr)} 샘플")
print(f"Test 데이터: {len(ym_te)} 샘플")
print(f"최종 6클래스 정확도: {hierarchical_acc:.4f}")

# 가중치 수렴 분석
if len(weight_history) > 10:
    recent_std = np.std(weight_history[-10:])
    print(f"최근 10 에포크 가중치 표준편차: {recent_std:.6f} (수렴도 지표)")
    
    if recent_std < 0.01:
        print("✅ 가중치가 안정적으로 수렴했습니다.")
    else:
        print("⚠️ 가중치가 아직 수렴 중이거나 진동하고 있습니다.")

print(f"\n=== Hessian 기반 증강된 데이터 학습 완료 ===")

# 최적화 방법별 비교를 위한 요약 정보 저장
optimization_summary = {
    'method': 'Hessian-based Adaptive Weight Optimization',
    'initial_weight': 0.3,
    'final_weight': model.get_current_weight(),
    'weight_change': abs(model.get_current_weight() - 0.3),
    'final_accuracy': hierarchical_acc,
    'convergence_std': np.std(weight_history[-10:]) if len(weight_history) > 10 else 0,
    'hessian_final': hessian_history[-1] if hessian_history else 0
}

print(f"\n최적화 요약:")
for key, value in optimization_summary.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.6f}")
    else:
        print(f"  {key}: {value}")

# Hessian 옵티마이저 설정값 출력
print(f"\n=== Hessian 옵티마이저 설정 ===")
if weight_optimizers:
    opt = weight_optimizers[0]
    print(f"학습률 (lr): {opt.lr}")
    print(f"Beta1 (모멘텀): {opt.beta1}")
    print(f"Beta2 (2차 모멘트): {opt.beta2}")
    print(f"Epsilon: {opt.eps}")
    print(f"Hessian 정규화: {opt.hessian_reg}")
    print(f"Gradient 히스토리 길이: {opt.max_history}")