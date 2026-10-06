import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 기본 설정
# ==========================================
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 연평균기온 선형회귀 예측")

st.write(
    "서울의 연평균기온 데이터를 이용하여 "
    "과거 데이터를 학습한 선형회귀 모델이 "
    "최근 20년의 기온을 얼마나 잘 예측하는지 평가합니다."
)


# ==========================================
# 데이터 주소
# ==========================================
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


# ==========================================
# 데이터 불러오기
# ==========================================
@st.cache_data
def load_data():
    return pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )


try:
    df = load_data()

except Exception as e:
    st.error("서울 기온 데이터를 불러오지 못했습니다.")
    st.error(str(e))
    st.stop()


# ==========================================
# 필요한 열 확인
# ==========================================
required_columns = [
    "날짜",
    "지점",
    "평균기온",
    "최저기온",
    "최고기온"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error("다음 열이 데이터에 없습니다.")
    st.write(missing_columns)
    st.stop()


# ==========================================
# 데이터 정리
# ==========================================
df["날짜"] = pd.to_datetime(
    df["날짜"],
    errors="coerce"
)

df["평균기온"] = pd.to_numeric(
    df["평균기온"],
    errors="coerce"
)

df = df.dropna(
    subset=["날짜", "평균기온"]
).copy()


# ==========================================
# 연도 추출
# ==========================================
df["연도"] = df["날짜"].dt.year


# ==========================================
# 2025년까지만 사용
# ==========================================
df = df[
    df["연도"] <= 2025
].copy()


# ==========================================
# 연도별 평균기온
# ==========================================
yearly = (
    df.groupby("연도")
    .agg(
        관측일수=("평균기온", "count"),
        연평균기온=("평균기온", "mean")
    )
    .reset_index()
)


# ==========================================
# 관측일 300일 미만인 연도 제외
# ==========================================
yearly = yearly[
    yearly["관측일수"] >= 300
].copy()


yearly = yearly.sort_values(
    "연도"
).reset_index(drop=True)


# ==========================================
# 분석에 필요한 기간 확인
# ==========================================
if yearly.empty:
    st.error("분석할 연도별 데이터가 없습니다.")
    st.stop()


# ==========================================
# 학습 / 테스트 데이터 분리
#
# 모델 1:
# 1956~2005 → 학습
#
# 모델 2:
# 1906~2005 → 학습
#
# 공통 테스트:
# 2006~2025
# ==========================================

train_50 = yearly[
    (yearly["연도"] >= 1956)
    & (yearly["연도"] <= 2005)
].copy()


train_100 = yearly[
    (yearly["연도"] >= 1906)
    & (yearly["연도"] <= 2005)
].copy()


test = yearly[
    (yearly["연도"] >= 2006)
    & (yearly["연도"] <= 2025)
].copy()


# ==========================================
# 데이터 존재 여부 확인
# ==========================================
if len(train_50) < 2:
    st.error(
        "1956~2005년 학습 데이터가 충분하지 않습니다."
    )
    st.stop()


if len(train_100) < 2:
    st.error(
        "1906~2005년 학습 데이터가 충분하지 않습니다."
    )
    st.stop()


if len(test) < 2:
    st.error(
        "2006~2025년 테스트 데이터가 충분하지 않습니다."
    )
    st.stop()


# ==========================================
# 독립변수 X
#
# 기존 분석과 동일하게
# 연도 - 1908을 사용
# ==========================================
train_50["지난연수"] = (
    train_50["연도"] - 1908
)

train_100["지난연수"] = (
    train_100["연도"] - 1908
)

test["지난연수"] = (
    test["연도"] - 1908
)


# ==========================================
# X, y 만들기
# ==========================================

X_train_50 = train_50[
    ["지난연수"]
]

y_train_50 = train_50[
    "연평균기온"
]

X_train_100 = train_100[
    ["지난연수"]
]

y_train_100 = train_100[
    "연평균기온"
]

X_test = test[
    ["지난연수"]
]

y_test = test[
    "연평균기온"
]


# ==========================================
# 선형회귀 모델 생성
# ==========================================

model_50 = LinearRegression()

model_100 = LinearRegression()


# ==========================================
# 모델 학습
# ==========================================

model_50.fit(
    X_train_50,
    y_train_50
)

model_100.fit(
    X_train_100,
    y_train_100
)


# ==========================================
# 테스트 데이터 예측
# ==========================================

pred_50 = model_50.predict(
    X_test
)

pred_100 = model_100.predict(
    X_test
)


# ==========================================
# 평가 함수
# ==========================================
def evaluate_model(
    y_true,
    y_pred
):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return mae, mse, r2


# ==========================================
# 모델 평가
# ==========================================

mae_50, mse_50, r2_50 = evaluate_model(
    y_test,
    pred_50
)

mae_100, mse_100, r2_100 = evaluate_model(
    y_test,
    pred_100
)


# ==========================================
# 기울기
# ==========================================

slope_50 = model_50.coef_[0]

slope_100 = model_100.coef_[0]


# ==========================================
# 100년당 기온 변화량
# ==========================================

slope_50_100 = slope_50 * 100

slope_100_100 = slope_100 * 100


# ==========================================
# 절편
# ==========================================

intercept_50 = model_50.intercept_

intercept_100 = model_100.intercept_


# ==========================================
# 제목
# ==========================================
st.subheader(
    "📚 학습 데이터와 테스트 데이터"
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "최근 50년 학습",
        f"{len(train_50)}개 연도"
    )

    st.write(
        "1956~2005년"
    )


with col2:

    st.metric(
        "최근 100년 학습",
        f"{len(train_100)}개 연도"
    )

    st.write(
        "1906~2005년"
    )


with col3:

    st.metric(
        "공통 테스트",
        f"{len(test)}개 연도"
    )

    st.write(
        "2006~2025년"
    )


# ==========================================
# 데이터 분할 설명
# ==========================================
st.info(
    "두 모델 모두 2006~2025년은 학습에 사용하지 않고 "
    "완전히 동일한 테스트 데이터로 성능을 평가합니다."
)


# ==========================================
# 회귀선 기울기 비교
# ==========================================
st.subheader(
    "📈 회귀선 기울기 비교"
)


col1, col2 = st.columns(2)


with col1:

    st.metric(
        "1956~2005 학습",
        f"{slope_50:+.4f} ℃ / 년"
    )

    st.write(
        f"100년당 "
        f"**{slope_50_100:+.2f} ℃**"
    )


with col2:

    st.metric(
        "1906~2005 학습",
        f"{slope_100:+.4f} ℃ / 년"
    )

    st.write(
        f"100년당 "
        f"**{slope_100_100:+.2f} ℃**"
    )


# ==========================================
# 회귀선 그래프
# ==========================================
st.subheader(
    "📊 학습기간에 따른 회귀선 비교"
)


graph_years = np.arange(
    1900,
    2026
)


graph_x = pd.DataFrame({
    "지난연수": graph_years - 1908
})


graph_pred_50 = model_50.predict(
    graph_x
)

graph_pred_100 = model_100.predict(
    graph_x
)


fig = go.Figure()


# 실제 데이터
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=7
        ),
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f} ℃"
            "<extra></extra>"
        )
    )
)


# 1956~2005 회귀선
fig.add_trace(
    go.Scatter(
        x=graph_years,
        y=graph_pred_50,
        mode="lines",
        name="1956~2005 학습 회귀선",
        line=dict(
            width=3
        ),
        hovertemplate=(
            "연도: %{x}년<br>"
            "50년 학습 회귀선: %{y:.2f} ℃"
            "<extra></extra>"
        )
    )
)


# 1906~2005 회귀선
fig.add_trace(
    go.Scatter(
        x=graph_years,
        y=graph_pred_100,
        mode="lines",
        name="1906~2005 학습 회귀선",
        line=dict(
            width=3,
            dash="dash"
        ),
        hovertemplate=(
            "연도: %{x}년<br>"
            "100년 학습 회귀선: %{y:.2f} ℃"
            "<extra></extra>"
        )
    )
)


# 테스트 기간 표시
fig.add_vrect(
    x0=2006,
    x1=2025,
    opacity=0.15,
    line_width=0,
    annotation_text="공통 테스트 기간",
    annotation_position="top left"
)


fig.update_layout(
    title="학습기간별 선형회귀선과 실제 연평균기온",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    xaxis=dict(
        range=[
            1900,
            2025
        ],
        tickmode="linear",
        dtick=10
    ),
    height=600,
    hovermode="x unified"
)


st.plotly_chart(
    fig,
    width="stretch"
)


# ==========================================
# 테스트 데이터 예측 결과
# ==========================================
st.subheader(
    "🎯 2006~2025년 테스트 데이터 예측"
)


result_df = test[
    ["연도", "연평균기온"]
].copy()


result_df["50년 학습 예측"] = pred_50

result_df["100년 학습 예측"] = pred_100

result_df["50년 오차"] = (
    result_df["연평균기온"]
    - result_df["50년 학습 예측"]
)

result_df["100년 오차"] = (
    result_df["연평균기온"]
    - result_df["100년 학습 예측"]
)


display_df = result_df.copy()


display_df["연평균기온"] = (
    display_df["연평균기온"]
    .round(2)
)

display_df["50년 학습 예측"] = (
    display_df["50년 학습 예측"]
    .round(2)
)

display_df["100년 학습 예측"] = (
    display_df["100년 학습 예측"]
    .round(2)
)

display_df["50년 오차"] = (
    display_df["50년 오차"]
    .round(2)
)

display_df["100년 오차"] = (
    display_df["100년 오차"]
    .round(2)
)


st.dataframe(
    display_df,
    width="stretch",
    hide_index=True
)


# ==========================================
# 테스트 성능 비교
# ==========================================
st.subheader(
    "🏆 테스트 데이터 예측 성능 비교"
)


comparison_df = pd.DataFrame({
    "모델": [
        "1956~2005 학습",
        "1906~2005 학습"
    ],

    "학습기간": [
        "1956~2005",
        "1906~2005"
    ],

    "기울기 (℃/년)": [
        slope_50,
        slope_100
    ],

    "기울기 (℃/100년)": [
        slope_50_100,
        slope_100_100
    ],

    "MAE (℃)": [
        mae_50,
        mae_100
    ],

    "MSE (℃²)": [
        mse_50,
        mse_100
    ],

    "R²": [
        r2_50,
        r2_100
    ]
})


display_comparison = comparison_df.copy()


display_comparison[
    "기울기 (℃/년)"
] = display_comparison[
    "기울기 (℃/년)"
].round(5)


display_comparison[
    "기울기 (℃/100년)"
] = display_comparison[
    "기울기 (℃/100년)"
].round(2)


display_comparison[
    "MAE (℃)"
] = display_comparison[
    "MAE (℃)"
].round(3)


display_comparison[
    "MSE (℃²)"
] = display_comparison[
    "MSE (℃²)"
].round(3)


display_comparison[
    "R²"
] = display_comparison[
    "R²"
].round(3)


st.dataframe(
    display_comparison,
    width="stretch",
    hide_index=True
)


# ==========================================
# 성능을 카드로 비교
# ==========================================
st.subheader(
    "📌 모델별 테스트 성능"
)


col1, col2 = st.columns(2)


with col1:

    st.markdown(
        "### 🔵 1956~2005년 학습"
    )

    st.metric(
        "MAE",
        f"{mae_50:.3f} ℃"
    )

    st.metric(
        "MSE",
        f"{mse_50:.3f} ℃²"
    )

    st.metric(
        "R²",
        f"{r2_50:.3f}"
    )


with col2:

    st.markdown(
        "### 🟠 1906~2005년 학습"
    )

    st.metric(
        "MAE",
        f"{mae_100:.3f} ℃"
    )

    st.metric(
        "MSE",
        f"{mse_100:.3f} ℃²"
    )

    st.metric(
        "R²",
        f"{r2_100:.3f}"
    )


# ==========================================
# 어느 모델이 더 좋은지 자동 판단
# ==========================================
st.subheader(
    "🔎 두 모델 비교 결과"
)


if mae_50 < mae_100:
    mae_result = (
        "MAE는 **1956~2005년 학습 모델**이 더 작아 "
        "평균적인 절대 예측 오차가 더 작습니다."
    )

elif mae_50 > mae_100:
    mae_result = (
        "MAE는 **1906~2005년 학습 모델**이 더 작아 "
        "평균적인 절대 예측 오차가 더 작습니다."
    )

else:
    mae_result = (
        "두 모델의 MAE가 같습니다."
    )


if mse_50 < mse_100:
    mse_result = (
        "MSE는 **1956~2005년 학습 모델**이 더 작습니다."
    )

elif mse_50 > mse_100:
    mse_result = (
        "MSE는 **1906~2005년 학습 모델**이 더 작습니다."
    )

else:
    mse_result = (
        "두 모델의 MSE가 같습니다."
    )


if r2_50 > r2_100:
    r2_result = (
        "R²는 **1956~2005년 학습 모델**이 더 높습니다."
    )

elif r2_50 < r2_100:
    r2_result = (
        "R²는 **1906~2005년 학습 모델**이 더 높습니다."
    )

else:
    r2_result = (
        "두 모델의 R²가 같습니다."
    )


st.write("• " + mae_result)

st.write("• " + mse_result)

st.write("• " + r2_result)


# ==========================================
# 기울기 차이
# ==========================================
slope_difference = (
    slope_50_100
    - slope_100_100
)


st.write(
    f"• 100년당 기온 변화량의 차이는 "
    f"**{slope_difference:+.2f} ℃**입니다."
)


# ==========================================
# 회귀식
# ==========================================
st.subheader(
    "📐 모델별 회귀식"
)


if intercept_50 >= 0:

    st.write(
        f"**1956~2005년 학습:** "
        f"연평균기온 = "
        f"{slope_50:.5f} × (연도 - 1908) "
        f"+ {intercept_50:.5f}"
    )

else:

    st.write(
        f"**1956~2005년 학습:** "
        f"연평균기온 = "
        f"{slope_50:.5f} × (연도 - 1908) "
        f"- {abs(intercept_50):.5f}"
    )


if intercept_100 >= 0:

    st.write(
        f"**1906~2005년 학습:** "
        f"연평균기온 = "
        f"{slope_100:.5f} × (연도 - 1908) "
        f"+ {intercept_100:.5f}"
    )

else:

    st.write(
        f"**1906~2005년 학습:** "
        f"연평균기온 = "
        f"{slope_100:.5f} × (연도 - 1908) "
        f"- {abs(intercept_100):.5f}"
    )


# ==========================================
# 데이터 처리 기준
# ==========================================
st.subheader(
    "📌 데이터 처리 기준"
)

st.write(
    "• 서울 연평균기온 데이터를 연도별 평균으로 변환했습니다."
)

st.write(
    "• 2025년 이후 데이터는 사용하지 않았습니다."
)

st.write(
    "• 연간 관측일수가 300일 미만인 연도는 제외했습니다."
)

st.write(
    "• 1956~2005년 데이터를 첫 번째 선형회귀 모델의 학습데이터로 사용했습니다."
)

st.write(
    "• 1906~2005년 데이터를 두 번째 선형회귀 모델의 학습데이터로 사용했습니다."
)

st.write(
    "• 2006~2025년은 두 모델이 공통으로 사용하지 않은 테스트 데이터입니다."
)

st.write(
    "• 두 모델은 동일한 2006~2025년 테스트 데이터로 비교했습니다."
)

st.write(
    "• 독립변수는 기존 분석과 동일하게 `연도 - 1908`을 사용했습니다."
)

st.write(
    "• MAE와 MSE는 작을수록 예측 오차가 작습니다."
)

st.write(
    "• R²는 일반적으로 1에 가까울수록 테스트 데이터를 잘 설명합니다."
)

st.write(
    "• 테스트 R²는 음수가 될 수도 있으며, 이는 단순히 테스트 데이터의 평균값을 사용하는 것보다도 예측이 좋지 않다는 의미가 될 수 있습니다."
)
