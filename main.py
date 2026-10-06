import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from numpy.polynomial import Polynomial

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="서울 연평균 기온 곡선 예측",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 연평균 기온 곡선 예측")

st.write(
    "연평균 기온 데이터를 이용하여 1차, 3차, 9차 다항회귀 모델을 만들고 "
    "학습에 사용하지 않은 테스트 데이터로 예측 성능을 평가합니다."
)

# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

try:
    df = pd.read_csv(DATA_URL, encoding="utf-8")
except Exception:
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

# 관측일수가 300일 이상인 연도만 사용
annual = annual[annual["관측일수"] >= 300].copy()

annual = annual.sort_values("연도").reset_index(drop=True)

if annual.empty:
    st.error("사용할 수 있는 연평균 기온 데이터가 없습니다.")
    st.stop()

# --------------------------------------------------
# 데이터 기간
# --------------------------------------------------

st.subheader("📊 연평균 기온 데이터")

st.write(
    f"분석에 사용한 연도: "
    f"{annual['연도'].min()}년 ~ {annual['연도'].max()}년"
)

# --------------------------------------------------
# 훈련 / 테스트 데이터 분리
# --------------------------------------------------

# 2005년 이전 → 훈련
train = annual[annual["연도"] < 2005].copy()

# 2005년부터 → 테스트
test = annual[annual["연도"] >= 2005].copy()

# --------------------------------------------------
# 훈련 / 테스트 개수 표시
# --------------------------------------------------

st.subheader("1. 훈련 데이터와 테스트 데이터")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "훈련 데이터 연도 수",
        f"{len(train)}개"
    )
    st.caption(
        f"{train['연도'].min()}년 ~ {train['연도'].max()}년"
        if len(train) > 0
        else "데이터 없음"
    )

with col2:
    st.metric(
        "테스트 데이터 연도 수",
        f"{len(test)}개"
    )
    st.caption(
        f"{test['연도'].min()}년 ~ {test['연도'].max()}년"
        if len(test) > 0
        else "데이터 없음"
    )

if len(train) < 10:
    st.error("훈련 데이터가 너무 적습니다.")
    st.stop()

if len(test) == 0:
    st.error("테스트 데이터가 없습니다.")
    st.stop()

# --------------------------------------------------
# 연도 크기 줄이기
# --------------------------------------------------

# 실제 연도를 그대로 사용하면 고차 다항식에서
# 숫자가 너무 커질 수 있으므로 기준 연도를 빼서 사용
기준연도 = train["연도"].mean()

x_train = train["연도"].to_numpy(dtype=float) - 기준연도
y_train = train["평균기온"].to_numpy(dtype=float)

x_test = test["연도"].to_numpy(dtype=float) - 기준연도
y_test = test["평균기온"].to_numpy(dtype=float)

# --------------------------------------------------
# 다항회귀 모델 만들기
# --------------------------------------------------

degrees = [1, 3, 9]

models = {}

for degree in degrees:
    model = Polynomial.fit(
        x_train,
        y_train,
        degree
    )

    models[degree] = model

# --------------------------------------------------
# 평가 함수
# --------------------------------------------------

def calculate_mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))

# --------------------------------------------------
# 테스트 데이터 예측 및 평가
# --------------------------------------------------

results = []

for degree in degrees:

    model = models[degree]

    # 테스트 데이터 예측
    test_prediction = model(x_test)

    # 테스트 데이터에서만 MAE 계산
    mae = calculate_mae(
        y_test,
        test_prediction
    )

    # 2050년 예측
    x_2050 = 2050 - 기준연도
    prediction_2050 = float(model(x_2050))

    results.append({
        "모델": f"{degree}차 곡선",
        "테스트 평균 오차 (MAE, ℃)": mae,
        "2050년 예측값 (℃)": prediction_2050
    })

results_df = pd.DataFrame(results)

# --------------------------------------------------
# 결과 표
# --------------------------------------------------

st.subheader("2. 곡선별 테스트 성능과 2050년 예측")

st.write(
    "세 모델 모두 2005년 이전의 훈련 데이터만 사용하여 학습했으며, "
    "MAE는 학습에 사용하지 않은 2005년 이후 테스트 데이터에서 계산했습니다."
)

st.dataframe(
    results_df.style.format({
        "테스트 평균 오차 (MAE, ℃)": "{:.3f}",
        "2050년 예측값 (℃)": "{:.2f}"
    }),
    use_container_width=True
)

# --------------------------------------------------
# 가장 좋은 모델
# --------------------------------------------------

best_index = results_df[
    "테스트 평균 오차 (MAE, ℃)"
].idxmin()

best_model_name = results_df.loc[
    best_index,
    "모델"
]

best_mae = results_df.loc[
    best_index,
    "테스트 평균 오차 (MAE, ℃)"
]

st.success(
    f"테스트 데이터에서 평균 오차가 가장 작은 모델은 "
    f"**{best_model_name}**이며, MAE는 **{best_mae:.3f}℃**입니다."
)

# --------------------------------------------------
# 2050년 예측값 카드
# --------------------------------------------------

st.subheader("3. 2050년 예상 연평균 기온")

col1, col2, col3 = st.columns(3)

for col, degree in zip(
    [col1, col2, col3],
    degrees
):

    prediction_2050 = results_df.loc[
        results_df["모델"] == f"{degree}차 곡선",
        "2050년 예측값 (℃)"
    ].iloc[0]

    with col:
        st.metric(
            f"{degree}차 곡선",
            f"{prediction_2050:.2f} ℃"
        )

# --------------------------------------------------
# 실제값 + 회귀곡선 그래프
# --------------------------------------------------

st.subheader("4. 훈련 데이터와 테스트 데이터, 곡선 비교")

fig = go.Figure()

# 훈련 데이터
fig.add_trace(
    go.Scatter(
        x=train["연도"],
        y=train["평균기온"],
        mode="markers",
        name="훈련 데이터"
    )
)

# 테스트 데이터
fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["평균기온"],
        mode="markers",
        name="테스트 데이터"
    )
)

# 곡선을 그릴 연도
curve_years = np.linspace(
    annual["연도"].min(),
    2050,
    500
)

curve_x = curve_years - 기준연도

# 1차, 3차, 9차 곡선
for degree in degrees:

    model = models[degree]

    curve_y = model(curve_x)

    fig.add_trace(
        go.Scatter(
            x=curve_years,
            y=curve_y,
            mode="lines",
            name=f"{degree}차 곡선"
        )
    )

# 2005년 경계선
fig.add_vline(
    x=2005,
    line_dash="dash",
    annotation_text="훈련 → 테스트"
)

# 2050년 경계선
fig.add_vline(
    x=2050,
    line_dash="dot",
    annotation_text="2050년"
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    height=600,
    hovermode="x unified"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# --------------------------------------------------
# 테스트 데이터 실제값과 예측값
# --------------------------------------------------

st.subheader("5. 테스트 데이터에서 실제값과 예측값 비교")

test_result = test[["연도", "평균기온"]].copy()

for degree in degrees:

    model = models[degree]

    test_result[
        f"{degree}차 예측값"
    ] = model(x_test)

    test_result[
        f"{degree}차 오차"
    ] = (
        test_result["평균기온"]
        - test_result[f"{degree}차 예측값"]
    )

st.dataframe(
    test_result.style.format({
        "평균기온": "{:.2f}",
        "1차 예측값": "{:.2f}",
        "1차 오차": "{:.2f}",
        "3차 예측값": "{:.2f}",
        "3차 오차": "{:.2f}",
        "9차 예측값": "{:.2f}",
        "9차 오차": "{:.2f}"
    }),
    use_container_width=True
)

# --------------------------------------------------
# 분석 방법 설명
# --------------------------------------------------

st.subheader("6. 분석 방법")

st.markdown(
    """
**① 훈련 데이터**

2005년 이전의 연평균 기온을 사용하여 1차, 3차, 9차 곡선을 학습했습니다.

**② 테스트 데이터**

2005년부터 2025년까지의 데이터를 학습 과정에 사용하지 않고,
마지막 평가 단계에서만 사용했습니다.

**③ MAE**

실제 기온과 모델이 예측한 기온의 차이를 절댓값으로 바꾼 뒤 평균을 계산했습니다.

따라서 **MAE가 작을수록 테스트 데이터를 더 잘 예측한 모델**입니다.

**④ 2050년 예측**

각 모델을 훈련 데이터에만 맞춘 뒤,
그 모델에 2050년을 입력하여 예측한 값입니다.

**⑤ 연도 변환**

고차 다항식을 계산할 때 연도 자체를 사용하면 숫자가 커질 수 있기 때문에
모든 연도에서 훈련 데이터의 평균 연도를 빼서 계산했습니다.

예를 들어,

`변환된 연도 = 실제 연도 - 훈련 데이터 평균 연도`

로 바꾸어 계산하고, 화면에는 다시 실제 연도로 표시합니다.
"""
)

# --------------------------------------------------
# 주의사항
# --------------------------------------------------

st.subheader("7. 해석할 때 주의할 점")

st.warning(
    "9차 곡선은 훈련 데이터의 복잡한 변화까지 따라갈 수 있지만, "
    "테스트 데이터에서 오히려 예측 오차가 커질 수도 있습니다. "
    "따라서 훈련 데이터에 얼마나 잘 맞는지만 보고 모델을 선택하지 않고, "
    "반드시 학습에 사용하지 않은 테스트 데이터의 MAE를 기준으로 비교해야 합니다."
)

st.caption(
    f"다항회귀 계산 기준연도: {기준연도:.2f}년 | "
    "연도별 관측일수 300일 이상인 자료만 사용"
)
