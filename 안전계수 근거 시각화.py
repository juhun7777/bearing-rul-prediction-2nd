import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.font_manager as fm

# 1) 사용할 한글 폰트 경로 지정 (윈도우: Malgun Gothic)
font_path = r"C:\Windows\Fonts\malgun.ttf"
font_prop = fm.FontProperties(fname=font_path).get_name()

# 한글 폰트 설정
plt.rcParams['font.family'] = font_prop  # Malgun Gothic 사용
plt.rcParams['axes.unicode_minus'] = False

def calculate_score(percent_error, E0=20, E1=50):
    """
    PHM 챌린지 스코어링 함수
    A_RUL = exp(-ln(0.5) × Er/E₀), if Er < 0 (보수적 예측)
    A_RUL = exp(+ln(0.5) × Er/E₀), if Er > 0 (낙관적 예측)
    """
    Er = percent_error
    
    if Er < 0:
        # 보수적 예측 (이른 예측) - 덜 가혹한 페널티
        score = np.exp(-np.log(0.5) * Er / E0)
    elif Er > 0:
        # 낙관적 예측 (늦은 예측) - 더 가혹한 페널티
        score = np.exp(np.log(0.5) * Er / E1)
    else:
        # 정확한 예측
        score = 1.0
    
    return score

# 기울기 계산 (페널티율 분석) - 더 정밀한 계산
# 여러 구간에서의 기울기 계산하여 평균값 구하기
conservative_slopes = []
optimistic_slopes = []

# 보수적 영역: 여러 구간에서 기울기 계산
for start in [-30, -25, -20, -15]:
    end = start + 10
    if end <= 0:
        slope = (calculate_score(end) - calculate_score(start)) / (end - start)
        conservative_slopes.append(abs(slope))

# 낙관적 영역: 여러 구간에서 기울기 계산  
for start in [0, 5, 10, 15]:
    end = start + 10
    if end <= 40:
        slope = (calculate_score(end) - calculate_score(start)) / (end - start)
        optimistic_slopes.append(abs(slope))

# 평균 기울기 계산
avg_conservative_slope = np.mean(conservative_slopes)
avg_optimistic_slope = np.mean(optimistic_slopes)

# 기울기 비율 (낙관적이 보수적보다 몇 배 더 가파른가)
slope_ratio = avg_optimistic_slope / avg_conservative_slope

# 스코어 손실률 계산 (1% 오차당 스코어 손실)
conservative_loss_rate = avg_conservative_slope * 100  # 1% 오차당 스코어 상승률
optimistic_loss_rate = avg_optimistic_slope * 100     # 1% 오차당 스코어 하락률

# 1. 연속적인 그래프 생성
percent_errors = np.linspace(-50, 50, 1000)
scores = [calculate_score(pe) for pe in percent_errors]

# 2. 특정 포인트들의 스코어 계산
specific_errors = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
specific_scores = [calculate_score(pe) for pe in specific_errors]

# 3. 시각화
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# 상단: 스코어링 함수 그래프
ax1.plot(percent_errors, scores, 'b-', linewidth=4, label='Scoring Function', alpha=0.8)
ax1.axvline(x=0, color='black', linestyle='--', alpha=0.7, linewidth=2, label='Perfect Prediction')

# 특정 포인트 표시 (더 큰 점)
ax1.scatter(specific_errors, specific_scores, color='red', s=120, zorder=5, 
           alpha=0.9, edgecolors='darkred', linewidth=2)

# 숫자 표시 개선 (교대로 높이 조정, 배경 추가)
for i, (pe, score) in enumerate(zip(specific_errors, specific_scores)):
    # 높이를 교대로 조정하여 겹치지 않게
    y_offset = 0.15 if i % 2 == 0 else 0.25
    y_pos = min(score + y_offset, 1.02)  # 그래프 범위를 벗어나지 않게
    
    # 배경이 있는 텍스트 박스
    ax1.annotate(f'{score:.3f}', 
                xy=(pe, score), 
                xytext=(pe, y_pos),
                ha='center', va='bottom',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                         edgecolor='red', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

# 축 설정
ax1.set_xlabel('Percent Error (%)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Score', fontsize=14, fontweight='bold')
ax1.set_title('PHM Challenge: Conservative Predictions Get Higher Scores', 
              fontsize=16, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.1)
ax1.set_xlim(-50, 50)

# 영역 색칠 (더 부드러운 색상)
ax1.axvspan(-50, 0, alpha=0.15, color='green', label='Conservative Zone')
ax1.axvspan(0, 50, alpha=0.15, color='red', label='Optimistic Zone')

# 영역 설명 텍스트 (위치 조정)
ax1.text(-25, 0.85, f'Conservative\n(Early Prediction)\n' +
         f'Score Gain: +{conservative_loss_rate:.3f}/1%\n' +
         f'Gentle Slope', 
         ha='center', va='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.9))

ax1.text(25, 0.35, f'Optimistic\n(Late Prediction)\n' +
         f'Score Loss: -{optimistic_loss_rate:.3f}/1%\n' +
         f'Steep Slope', 
         ha='center', va='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral', alpha=0.9))

ax1.legend(fontsize=12, loc='upper right')

# 하단: 테이블 형태의 시각화 (간단한 버전)
# 데이터프레임 생성
df = pd.DataFrame({
    'Percent Error (%)': specific_errors,
    'Score': [f'{score:.4f} ({"보수적" if pe < 0 else "정확" if pe == 0 else "낙관적"})' 
              for pe, score in zip(specific_errors, specific_scores)]
})

# 테이블을 이미지로 만들기
ax2.axis('tight')
ax2.axis('off')

# 색상 맵핑 - 각 행에 대한 색상 배열 생성
row_colors = []
for pe in specific_errors:
    if pe < 0:
        row_color = ['#E8F5E8', '#E8F5E8']  # 연한 초록 (보수적) - 2개 컬럼
    elif pe == 0:
        row_color = ['#F0F8FF', '#F0F8FF']  # 연한 파랑 (정확) - 2개 컬럼
    else:
        row_color = ['#FFE8E8', '#FFE8E8']  # 연한 빨강 (낙관적) - 2개 컬럼
    row_colors.append(row_color)

table = ax2.table(cellText=df.values,
                 colLabels=df.columns,
                 cellLoc='center',
                 loc='center',
                 cellColours=row_colors)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2)

# 헤더 스타일링
for i in range(len(df.columns)):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax2.set_title('Score Values at Different Prediction Errors', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()

# 분석 결과 출력
print("=" * 70)
print("              PHM Challenge Scoring Function Analysis")
print("=" * 70)

print("🔍 Key Insights:")
print("-" * 50)
print("✅ Conservative predictions (early) receive HIGHER scores")
print("❌ Optimistic predictions (late) receive LOWER scores")
print("🎯 Perfect prediction (0% error) gets maximum score of 1.0")
print(f"📈 Detailed Slope Analysis:")
print(f"   Conservative slope (average): {avg_conservative_slope:.4f} score/1% error")
print(f"   Optimistic slope (average):   {avg_optimistic_slope:.4f} score/1% error")
print(f"   Slope ratio:                  {slope_ratio:.2f}× steeper for late predictions")
print(f"   Penalty severity difference:  {((slope_ratio-1)*100):.0f}% more severe")

print(f"\n💡 Why Early Prediction Matters:")
print("-" * 50)
print(f"• Mathematical advantage: {slope_ratio:.1f}× gentler penalty slope")
print(f"• Score gain rate: +{conservative_loss_rate:.3f} per 1% early prediction")
print(f"• Score loss rate: -{optimistic_loss_rate:.3f} per 1% late prediction")
print(f"• Risk asymmetry: Late errors are {((slope_ratio-1)*100):.0f}% more costly")
print("• Safety benefit: Prevents unexpected failures")
print("• Cost predictability: Planned maintenance vs emergency repairs")

print(f"\n📊 Comparison at ±20% Error:")
print(f"   Conservative (-20%): {calculate_score(-20):.4f}")
print(f"   Optimistic (+20%):   {calculate_score(20):.4f}")
print(f"   Score advantage:     {calculate_score(-20) - calculate_score(20):.4f} ({((calculate_score(-20) - calculate_score(20))/calculate_score(20)*100):.1f}%)")
print(f"   Slope impact:        Early gains {conservative_loss_rate*20:.3f}, Late loses {optimistic_loss_rate*20:.3f}")

print(f"\n📊 Comparison at ±30% Error:")
print(f"   Conservative (-30%): {calculate_score(-30):.4f}")
print(f"   Optimistic (+30%):   {calculate_score(30):.4f}")
print(f"   Score advantage:     {calculate_score(-30) - calculate_score(30):.4f} ({((calculate_score(-30) - calculate_score(30))/calculate_score(30)*100):.1f}%)")
print(f"   Slope impact:        Early gains {conservative_loss_rate*30:.3f}, Late loses {optimistic_loss_rate*30:.3f}")

print("\n🏭 Industrial Implications:")
print("-" * 50)
print("• Early replacement → Safe but costly")
print("• Late replacement → Risk of failure")
print("• PHM scoring favors safety over cost optimization")
print("• Encourages conservative maintenance strategies")

plt.show()

# 4. 상세 스코어 테이블 (원본 코드 방식)
print("\n" + "=" * 70)
print("                    Detailed Score Table")
print("=" * 70)
detailed_df = pd.DataFrame({
    'Percent Error (%)': specific_errors,
    'Score': specific_scores,
    'Score (4자리)': [f'{score:.4f}' for score in specific_scores],
    'Prediction Type': ['Conservative' if pe < 0 else 'Accurate' if pe == 0 else 'Optimistic' 
                       for pe in specific_errors],
    'Safety Implication': ['Early replacement (Safe)' if pe < 0 else 'Perfect timing' if pe == 0 
                          else 'Late replacement (Risk)' for pe in specific_errors]
})

print(detailed_df.to_string(index=False, justify='center'))