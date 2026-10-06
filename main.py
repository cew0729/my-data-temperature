import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="기온 선형회귀 모델 평가",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 연평균 기온 선형회귀 모델 평가")

st.write(
    "서울의 연평균 기온 데이터를 이용하여 선형회귀 모델을 만들고, "
    "50년·100년의 과거 데이터를 학습한 모델이 최근 20년의 기온을 "
    "얼마나 잘 예측하는지 비교합니다."
)

# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

try:
    df = pd.read_csv(DATA_URL, encoding="utf-8")
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.stop()

# --------------------------------------------------
# 데이터 전처리
# --------------------------------------------------

df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

df = df.dropna(subset=["날짜", "평균기온"])

df["연도"] = df["날짜"].dt.year

# 2025년까지 사용
df = df[df["연도"] <= 2025]

# --------------------------------------------------
# 연평균 기온 계산
# --------------------------------------------------

annual = (
    df.groupby("연도")
    .agg(
        평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)

# 1년의 관측값이 300일보다 적은 연도는 제외
annual = annual[annual["관측일수"] >= 300].copy()

annual = annual.sort_values("연도").reset_index(drop=True)

# --------------------------------------------------
# 전체 데이터 확인
# --------------------------------------------------

if annual.empty:
    st.error("사용할 수 있는 연평균 기온 데이터가 없습니다.")
    st.stop()

st.subheader("1. 연평균 기온 데이터")

st.write(
    f"관측일수가 300일 이상인 연도만 사용했습니다. "
    f"사용 가능한 연도는 {annual['연도'].min()}년부터 "
    f"{annual['연도'].max()}년까지입니다."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("사용 연도 수", f"{len(annual)}년")

with col2:
    st.metric("시작 연도", f"{annual['연도'].min()}년")

with col3:
    st.metric("마지막 연도", f"{annual['연도'].max()}년")

# --------------------------------------------------
# 선형회귀 함수
# --------------------------------------------------

def linear_regression(data):
    """
    최소제곱법을 이용한 단순 선형회귀
    y = slope * x + intercept
    """

    x = data["연도"].to_numpy(dtype=float) - 1908
    y = data["평균기온"].to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, 1)

    prediction = slope * x + intercept

    return slope, intercept, prediction


# --------------------------------------------------
# 평가 지표 함수
# --------------------------------------------------

def evaluate_model(y_true, y_pred):
    """
    MAE, MSE, R² 계산
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    error = y_true - y_pred

    mae = np.mean(np.abs(error))
    mse = np.mean(error ** 2)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        r2 = np.nan
    else:
        r2 = 1 - (ss_res / ss_tot)

    return mae, mse, r2


# --------------------------------------------------
# 2. 전체 데이터에 대한 선형회귀
# --------------------------------------------------

st.subheader("2. 전체 데이터에 대한 선형회귀")

full_slope, full_intercept, full_pred = linear_regression(annual)

full_mae, full_mse, full_r2 = evaluate_model(
    annual["평균기온"],
    full_pred
)

st.write(
    "전체 연평균 기온 데이터를 이용하여 하나의 선형회귀선을 만들었습니다."
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "전체 데이터 기울기",
        f"{full_slope:.4f} ℃/년"
    )

with col2:
    st.metric(
        "100년당 변화",
        f"{full_slope * 100:.2f} ℃"
    )

with col3:
    st.metric(
        "전체 데이터 MAE",
        f"{full_mae:.3f} ℃"
    )

with col4:
    st.metric(
        "전체 데이터 R²",
        f"{full_r2:.3f}"
    )

st.write(
    f"**전체 데이터 회귀식:** "
    f"평균기온 = {full_slope:.4f} × (연도 - 1908) "
    f"+ {full_intercept:.4f}"
)

# --------------------------------------------------
# 3. 훈련 데이터와 테스트 데이터 분리
# --------------------------------------------------

st.subheader("3. 훈련 데이터와 테스트 데이터")

train_50 = annual[
    (annual["연도"] >= 1956) &
    (annual["연도"] <= 2005)
].copy()

train_100 = annual[
    (annual["연도"] >= 1906) &
    (annual["연도"] <= 2005)
].copy()

test = annual[
    (annual["연도"] >= 2006) &
    (annual["연도"] <= 2025)
].copy()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "50년 훈련 데이터",
        f"{len(train_50)}년"
    )
    st.caption("1956~2005")

with col2:
    st.metric(
        "100년 훈련 데이터",
        f"{len(train_100)}년"
    )
    st.caption("1906~2005")

with col3:
    st.metric(
        "공통 테스트 데이터",
        f"{len(test)}년"
    )
    st.caption("2006~2025")

# --------------------------------------------------
# 데이터 부족 여부 확인
# --------------------------------------------------

if len(train_50) == 0:
    st.error("1956~2005년의 훈련 데이터가 없습니다.")
    st.stop()

if len(train_100) == 0:
    st.error("1906~2005년의 훈련 데이터가 없습니다.")
    st.stop()

if len(test) == 0:
    st.error("2006~2025년의 테스트 데이터가 없습니다.")
    st.stop()

# --------------------------------------------------
# 4. 50년 훈련 모델
# --------------------------------------------------

slope_50, intercept_50, train_pred_50 = linear_regression(train_50)

# 테스트 데이터 예측
x_test = test["연도"].to_numpy(dtype=float) - 1908
y_test = test["평균기온"].to_numpy(dtype=float)

pred_50 = slope_50 * x_test + intercept_50

mae_50, mse_50, r2_50 = evaluate_model(
    y_test,
    pred_50
)

# --------------------------------------------------
# 5. 100년 훈련 모델
# --------------------------------------------------

slope_100, intercept_100, train_pred_100 = linear_regression(train_100)

# 테스트 데이터 예측
pred_100 = slope_100 * x_test + intercept_100

mae_100, mse_100, r2_100 = evaluate_model(
    y_test,
    pred_100
)

# --------------------------------------------------
# 6. 회귀선 기울기 비교
# --------------------------------------------------

st.subheader("4. 50년 학습과 100년 학습의 회귀선 비교")

comparison_slope = pd.DataFrame({
    "학습 기간": [
        "최근 50년 (1956~2005)",
        "최근 100년 (1906~2005)"
    ],
    "기울기 (℃/년)": [
        slope_50,
        slope_100
    ],
    "100년당 변화 (℃)": [
        slope_50 * 100,
        slope_100 * 100
    ]
})

st.dataframe(
    comparison_slope.style.format({
        "기울기 (℃/년)": "{:.4f}",
        "100년당 변화 (℃)": "{:.2f}"
    }),
    use_container_width=True
)

# --------------------------------------------------
# 7. 테스트 데이터 성능 비교
# --------------------------------------------------

st.subheader("5. 최근 20년 테스트 데이터 예측 성능")

comparison = pd.DataFrame({
    "모델": [
        "50년 학습 모델",
        "100년 학습 모델"
    ],
    "훈련 기간": [
        "1956~2005",
        "1906~2005"
    ],
    "MAE": [
        mae_50,
        mae_100
    ],
    "MSE": [
        mse_50,
        mse_100
    ],
    "R²": [
        r2_50,
        r2_100
    ]
})

st.dataframe(
    comparison.style.format({
        "MAE": "{:.3f}",
        "MSE": "{:.3f}",
        "R²": "{:.3f}"
    }),
    use_container_width=True
)

# --------------------------------------------------
# 성능 카드
# --------------------------------------------------

st.markdown("### 📊 모델별 테스트 성능")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🟦 50년 학습 모델")

    st.metric("MAE", f"{mae_50:.3f} ℃")
    st.metric("MSE", f"{mse_50:.3f}")
    st.metric("R²", f"{r2_50:.3f}")

with col2:
    st.markdown("#### 🟩 100년 학습 모델")

    st.metric("MAE", f"{mae_100:.3f} ℃")
    st.metric("MSE", f"{mse_100:.3f}")
    st.metric("R²", f"{r2_100:.3f}")

# --------------------------------------------------
# 8. 어떤 모델이 더 좋은지 자동 비교
# --------------------------------------------------

st.subheader("6. 두 모델의 예측 성능 비교")

if mae_50 < mae_100:
    mae_result = "50년 학습 모델의 MAE가 더 낮아 평균적인 예측 오차가 작습니다."
elif mae_100 < mae_50:
    mae_result = "100년 학습 모델의 MAE가 더 낮아 평균적인 예측 오차가 작습니다."
else:
    mae_result = "두 모델의 MAE가 같습니다."

if mse_50 < mse_100:
    mse_result = "50년 학습 모델의 MSE가 더 낮습니다."
elif mse_100 < mse_50:
    mse_result = "100년 학습 모델의 MSE가 더 낮습니다."
else:
    mse_result = "두 모델의 MSE가 같습니다."

if r2_50 > r2_100:
    r2_result = "50년 학습 모델의 R²가 더 높아 테스트 데이터의 변동을 더 잘 설명합니다."
elif r2_100 > r2_50:
    r2_result = "100년 학습 모델의 R²가 더 높아 테스트 데이터의 변동을 더 잘 설명합니다."
else:
    r2_result = "두 모델의 R²가 같습니다."

st.write("**MAE 비교:**", mae_result)
st.write("**MSE 비교:**", mse_result)
st.write("**R² 비교:**", r2_result)

# --------------------------------------------------
# 9. 실제값과 예측값 비교 그래프
# --------------------------------------------------

st.subheader("7. 실제 연평균 기온과 회귀선 비교")

fig = go.Figure()

# 실제 연평균 기온
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["평균기온"],
        mode="lines+markers",
        name="실제 연평균 기온"
    )
)

# 50년 회귀선
years_50 = np.arange(1956, 2026)
x_50 = years_50 - 1908
line_50 = slope_50 * x_50 + intercept_50

fig.add_trace(
    go.Scatter(
        x=years_50,
        y=line_50,
        mode="lines",
        name="50년 학습 회귀선"
    )
)

# 100년 회귀선
years_100 = np.arange(1906, 2026)
x_100 = years_100 - 1908
line_100 = slope_100 * x_100 + intercept_100

fig.add_trace(
    go.Scatter(
        x=years_100,
        y=line_100,
        mode="lines",
        name="100년 학습 회귀선"
    )
)

# 테스트 기간 표시
fig.add_vrect(
    x0=2006,
    x1=2025,
    fillcolor="gray",
    opacity=0.15,
    line_width=0,
    annotation_text="공통 테스트 기간",
    annotation_position="top left"
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="평균기온 (℃)",
    hovermode="x unified",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# 10. 테스트 기간 실제값과 예측값
# --------------------------------------------------

st.subheader("8. 최근 20년 실제값과 예측값")

result_test = test[["연도", "평균기온"]].copy()

result_test["50년 모델 예측"] = pred_50
result_test["100년 모델 예측"] = pred_100

result_test["50년 모델 오차"] = (
    result_test["평균기온"] -
    result_test["50년 모델 예측"]
)

result_test["100년 모델 오차"] = (
    result_test["평균기온"] -
    result_test["100년 모델 예측"]
)

st.dataframe(
    result_test.style.format({
        "평균기온": "{:.2f}",
        "50년 모델 예측": "{:.2f}",
        "100년 모델 예측": "{:.2f}",
        "50년 모델 오차": "{:.2f}",
        "100년 모델 오차": "{:.2f}"
    }),
    use_container_width=True
)

# --------------------------------------------------
# 11. 회귀식
# --------------------------------------------------

st.subheader("9. 만들어진 회귀식")

st.write(
    f"**50년 학습 회귀식**  \n"
    f"평균기온 = {slope_50:.4f} × (연도 - 1908) + {intercept_50:.4f}"
)

st.write(
    f"**100년 학습 회귀식**  \n"
    f"평균기온 = {slope_100:.4f} × (연도 - 1908) + {intercept_100:.4f}"
)

# --------------------------------------------------
# 12. 분석 결과 해석
# --------------------------------------------------

st.subheader("10. 분석 결과 해석")

st.markdown(
    """
### 🔎 분석 방법

- **50년 모델:** 1956~2005년 데이터를 훈련 데이터로 사용
- **100년 모델:** 1906~2005년 데이터를 훈련 데이터로 사용
- **테스트 데이터:** 2006~2025년을 두 모델에서 공통으로 사용
- 두 모델이 동일한 최근 20년 데이터를 예측하도록 하여 공정하게 비교
- **MAE:** 평균적으로 실제 기온에서 얼마나 벗어났는지 나타냄
- **MSE:** 큰 예측 오차에 더 큰 패널티를 주어 평가
- **R²:** 실제 기온의 변동을 회귀모델이 얼마나 설명하는지 나타냄

### 📌 지표 해석

- **MAE → 낮을수록 좋음**
- **MSE → 낮을수록 좋음**
- **R² → 높을수록 좋음**

따라서 최근 20년을 예측할 때는  
MAE와 MSE가 더 낮고 R²가 더 높은 모델이 더 좋은 예측 성능을 가진 것으로 판단할 수 있습니다.
"""
)

# --------------------------------------------------
# 13. 데이터 처리 기준
# --------------------------------------------------

st.subheader("11. 데이터 처리 기준")

st.write(
    "원본 일별 기온 데이터에서 날짜와 평균기온이 정상적으로 기록된 자료를 사용하고, "
    "연도별 평균을 계산했습니다. 또한 한 해의 관측일수가 300일 미만인 연도는 "
    "연평균 계산에서 제외했습니다."
)

st.write(
    f"현재 분석에 사용된 연도 범위: "
    f"{annual['연도'].min()}~{annual['연도'].max()}"
)
