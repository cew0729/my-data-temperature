import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# ==========================================
# 기본 설정
# ==========================================
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")

st.write(
    "서울의 연평균기온 데이터를 이용하여 "
    "기온 변화 추세를 분석하고 미래 기온을 예측합니다."
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
# 2025년까지의 데이터만 사용
# ==========================================
df = df[
    df["연도"] <= 2025
].copy()


# ==========================================
# 연도별 평균기온 계산
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


# ==========================================
# 연도순 정렬
# ==========================================
yearly = yearly.sort_values(
    "연도"
).reset_index(drop=True)


# ==========================================
# 회귀 가능한지 확인
# ==========================================
if len(yearly) < 2:
    st.error(
        "회귀분석을 수행할 수 있는 데이터가 충분하지 않습니다."
    )
    st.stop()


# ==========================================
# 독립변수
# 1908년부터 지난 연수
# ==========================================
yearly["지난연수"] = (
    yearly["연도"] - 1908
)


# ==========================================
# 전체 기간 회귀
# ==========================================
x = yearly["지난연수"].to_numpy(
    dtype=float
)

y = yearly["연평균기온"].to_numpy(
    dtype=float
)

slope, intercept = np.polyfit(
    x,
    y,
    1
)


# ==========================================
# 전체 기간 상관계수
# ==========================================
correlation = np.corrcoef(
    x,
    y
)[0, 1]


# ==========================================
# 전체 기간 100년당 변화량
# ==========================================
slope_100 = slope * 100


# ==========================================
# 전체 회귀 기간 정보
# ==========================================
number_of_years = len(yearly)

start_year = int(
    yearly["연도"].min()
)

end_year = int(
    yearly["연도"].max()
)


# ==========================================
# 최근 20년 데이터
# ==========================================
recent_start_year = end_year - 19

recent_20 = yearly[
    yearly["연도"] >= recent_start_year
].copy()


if len(recent_20) >= 2:

    recent_x = recent_20[
        "지난연수"
    ].to_numpy(
        dtype=float
    )

    recent_y = recent_20[
        "연평균기온"
    ].to_numpy(
        dtype=float
    )

    # 최근 20년 회귀
    recent_slope, recent_intercept = np.polyfit(
        recent_x,
        recent_y,
        1
    )

    # 최근 20년 상관계수
    recent_correlation = np.corrcoef(
        recent_x,
        recent_y
    )[0, 1]

    # 최근 20년 100년당 변화량
    recent_slope_100 = recent_slope * 100

else:

    recent_slope = np.nan
    recent_intercept = np.nan
    recent_correlation = np.nan
    recent_slope_100 = np.nan


# ==========================================
# 예측 함수
# ==========================================
def predict_temperature(year):
    return (
        slope * (year - 1908)
        + intercept
    )


# ==========================================
# 회귀선용 연도
# ==========================================
prediction_years = np.arange(
    1900,
    2101
)


# 전체 기간 회귀선
prediction_temperatures = (
    slope * (prediction_years - 1908)
    + intercept
)


# 최근 20년 회귀선
if not np.isnan(recent_slope):

    recent_prediction_temperatures = (
        recent_slope
        * (prediction_years - 1908)
        + recent_intercept
    )


# ==========================================
# 100년당 기온 변화
# ==========================================
st.subheader(
    "🌡️ 100년에 기온이 얼마나 변하는가?"
)

col1, col2 = st.columns(2)


with col1:

    st.metric(
        label="전체 기간",
        value=f"{slope_100:+.2f} ℃ / 100년"
    )


with col2:

    if not np.isnan(recent_slope_100):

        st.metric(
            label=(
                f"최근 20년 "
                f"({recent_start_year}~{end_year})"
            ),
            value=(
                f"{recent_slope_100:+.2f} ℃ / 100년"
            )
        )

    else:

        st.metric(
            label="최근 20년",
            value="계산 불가"
        )


# ==========================================
# 산점도와 회귀선
# ==========================================
st.subheader(
    "📈 서울 연평균기온과 회귀선"
)


fig = go.Figure()


# 실제 연평균기온
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=7
        ),
        customdata=yearly[
            ["관측일수"]
        ],
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f} ℃<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        )
    )
)


# 전체 기간 회귀선
fig.add_trace(
    go.Scatter(
        x=prediction_years,
        y=prediction_temperatures,
        mode="lines",
        name="전체 기간 회귀선",
        line=dict(
            width=3
        ),
        hovertemplate=(
            "연도: %{x}년<br>"
            "전체 기간 예상기온: %{y:.2f} ℃"
            "<extra></extra>"
        )
    )
)


# 최근 20년 회귀선
if not np.isnan(recent_slope):

    fig.add_trace(
        go.Scatter(
            x=prediction_years,
            y=recent_prediction_temperatures,
            mode="lines",
            name="최근 20년 회귀선",
            line=dict(
                width=3,
                dash="dash"
            ),
            hovertemplate=(
                "연도: %{x}년<br>"
                "최근 20년 회귀선: %{y:.2f} ℃"
                "<extra></extra>"
            )
        )
    )


# 그래프 설정
fig.update_layout(
    title="서울 연평균기온과 회귀선 비교",

    xaxis_title="연도",

    yaxis_title="연평균기온 (℃)",

    xaxis=dict(
        tickmode="linear",
        dtick=10,
        range=[
            1900,
            2100
        ]
    ),

    hovermode="x unified",

    height=600,

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)


st.plotly_chart(
    fig,
    width="stretch"
)


# ==========================================
# 회귀 분석 정보
# ==========================================
st.subheader(
    "📊 회귀 분석 정보"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "회귀에 사용된 연도",
        f"{number_of_years}개"
    )


with col2:

    st.metric(
        "시작 연도",
        f"{start_year}년"
    )


with col3:

    st.metric(
        "끝 연도",
        f"{end_year}년"
    )


with col4:

    st.metric(
        "전체 기간 상관계수",
        f"{correlation:.3f}"
    )


# ==========================================
# 회귀식
# ==========================================
st.subheader(
    "📐 회귀식"
)


if intercept >= 0:

    st.write(
        f"**전체 기간:** "
        f"연평균기온 = "
        f"{slope:.5f} × (연도 - 1908) "
        f"+ {intercept:.5f}"
    )

else:

    st.write(
        f"**전체 기간:** "
        f"연평균기온 = "
        f"{slope:.5f} × (연도 - 1908) "
        f"- {abs(intercept):.5f}"
    )


if not np.isnan(recent_slope):

    if recent_intercept >= 0:

        st.write(
            f"**최근 20년:** "
            f"연평균기온 = "
            f"{recent_slope:.5f} × (연도 - 1908) "
            f"+ {recent_intercept:.5f}"
        )

    else:

        st.write(
            f"**최근 20년:** "
            f"연평균기온 = "
            f"{recent_slope:.5f} × (연도 - 1908) "
            f"- {abs(recent_intercept):.5f}"
        )


# ==========================================
# 상관계수 비교
# ==========================================
st.subheader(
    "🔗 상관계수 비교"
)


col1, col2 = st.columns(2)


with col1:

    st.metric(
        "전체 기간",
        f"{correlation:.3f}"
    )


with col2:

    if not np.isnan(recent_correlation):

        st.metric(
            "최근 20년",
            f"{recent_correlation:.3f}"
        )

    else:

        st.metric(
            "최근 20년",
            "계산 불가"
        )


# ==========================================
# 기온 예측
# ==========================================
st.subheader(
    "🔮 기온 예측하기"
)


selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)


# ==========================================
# 선택한 연도의 예상기온
# ==========================================
predicted_temperature = predict_temperature(
    selected_year
)


# ==========================================
# 예상기온 표시
# ==========================================
st.metric(
    label=f"{selected_year}년 예상 연평균기온",
    value=f"{predicted_temperature:.2f} ℃"
)


# ==========================================
# 선택한 연도 위치 그래프
# ==========================================
selected_fig = go.Figure()


# 실제 연평균기온
selected_fig.add_trace(
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


# 전체 기간 회귀선
selected_fig.add_trace(
    go.Scatter(
        x=prediction_years,
        y=prediction_temperatures,
        mode="lines",
        name="전체 기간 회귀선",
        line=dict(
            width=3
        ),
        hovertemplate=(
            "연도: %{x}년<br>"
            "예상기온: %{y:.2f} ℃"
            "<extra></extra>"
        )
    )
)


# 최근 20년 회귀선
if not np.isnan(recent_slope):

    selected_fig.add_trace(
        go.Scatter(
            x=prediction_years,
            y=recent_prediction_temperatures,
            mode="lines",
            name="최근 20년 회귀선",
            line=dict(
                width=3,
                dash="dash"
            ),
            hovertemplate=(
                "연도: %{x}년<br>"
                "최근 20년 회귀선: %{y:.2f} ℃"
                "<extra></extra>"
            )
        )
    )


# 선택한 연도의 예상값
selected_fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temperature],
        mode="markers",
        name=f"{selected_year}년 예상값",
        marker=dict(
            size=15,
            symbol="diamond"
        ),
        hovertemplate=(
            f"{selected_year}년<br>"
            f"예상기온: {predicted_temperature:.2f} ℃"
            "<extra></extra>"
        )
    )
)


# 그래프 설정
selected_fig.update_layout(
    title=f"{selected_year}년 예상기온 위치",

    xaxis_title="연도",

    yaxis_title="연평균기온 (℃)",

    xaxis=dict(
        tickmode="linear",
        dtick=10,
        range=[
            1900,
            2100
        ]
    ),

    height=550,

    hovermode="x unified"
)


st.plotly_chart(
    selected_fig,
    width="stretch"
)


# ==========================================
# 데이터 처리 기준
# ==========================================
st.subheader(
    "📌 데이터 처리 기준"
)

st.write(
    "• 2025년까지의 데이터만 사용했습니다."
)

st.write(
    "• 평균기온이 기록된 관측일을 이용하여 "
    "연도별 평균기온을 계산했습니다."
)

st.write(
    "• 관측일수가 300일 미만인 연도는 제외했습니다."
)

st.write(
    "• 회귀분석의 독립변수는 "
    "`연도 - 1908`입니다."
)

st.write(
    f"• 최근 20년 회귀는 "
    f"{recent_start_year}년부터 {end_year}년까지 "
    "사용 가능한 자료를 이용했습니다."
)

st.write(
    "• 100년당 기온 변화량은 "
    "회귀선의 연간 기울기에 100을 곱하여 계산했습니다."
)

st.write(
    "• 1900~2100년 예상기온은 "
    "회귀식을 이용한 계산값이며 실제 관측값이 아닙니다."
)
