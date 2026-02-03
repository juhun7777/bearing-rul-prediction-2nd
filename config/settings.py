"""
전역 설정 및 상수 정의
"""
import os

# ========================================
# 전역 설정
# ========================================
INTERVAL_MIN = 10    # 10분 간격
DURATION_S = 10      # 10초 측정
FS = 25600.0         # 샘플링 주파수 (Hz)

# ========================================
# 경로 설정 
# ========================================


# === 설정 ===
BASE_DIR        = "/data"  # 대회 환경 기준
TRAIN_DIR       = os.path.join(BASE_DIR, "Train")
VALIDATION_DIR  = os.path.join(BASE_DIR, "Validation Set")
MODEL_DIR = "./saved_models"


print("Using actual data directory structure")
print(f"VALIDATION_DIR: {VALIDATION_DIR}")
print(f"TRAIN_DIR: {TRAIN_DIR}")
print(f"MODEL_DIR: {MODEL_DIR}")

# 경로 존재 확인
if os.path.exists(VALIDATION_DIR):
    print("✅ Validation directory found")
else:
    print("❌ Validation directory not found!")

if os.path.exists(TRAIN_DIR):
    print("✅ Train directory found")
else:
    print("❌ Train directory not found!")
