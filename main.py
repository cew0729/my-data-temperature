import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# -----------------------------------
# 기본 설정
# -----------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write("서울의 연평균기온 데이터를 이용하여 회귀 직선으로 기온을 예측합니다.")


# -----------------------------------
# 데이터 주소
# -----------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


# -----------------------------------
# 데이터 불러오기
# -----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df


try:
    df = load_data()

except Exception as e:
    st.error("서울 기온 데이터를 불러오지 못했습니다.")
    st.error(f"오류 내용: {e}")
    st.stop()


# -----------------------------------
# 필요한 열 확인
# -----------------------------------
required_columns = [
    "날짜",
    "지점",
    "평균기온",
    "최저기온",
    "최고기온"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error("다음 열이 데이터에 없습니다.")
    st.write(missing_columns)
    st.stop()


# -----------------------------------
# 날짜 및 기온 데이터 정리
# -----------------------------------
df["날짜"] = pd.to_datetime(
    df["날짜"],
    errors="coerce"
)

df["평균기온"] = pd.to_numeric(
    df["평균기온"],
    errors="coerce"
)

# 날짜와 평균기온이 모두 있는 자료만 사용
df = df.dropna(
    subset=["날짜", "평균기온"]
).copy()

# 연도 추출
df["연도"] = df["날짜"].dt.year


# -----------------------------------
# 2025년까지의 자료만 사용
# -----------------------------------
df = df[df["연도"] <= 2025].copy()


# -----------------------------------
# 연도별 관측일 수와 연평균기온 계산
# -----------------------------------
yearly = (
    df.groupby("연도")
    .agg(
        관측일수=("평균기온", "count"),
        연평균기온=("평균기온", "mean")
    )
    .reset_index()
)


# -----------------------------------
# 관측일이 300일 이상인 해만 사용
# -----------------------------------
yearly = yearly[
    yearly["관측일수"] >= 300
].copy()


# 연도순 정렬
yearly = yearly.sort_values("연도").reset_index(drop=True)


# -----------------------------------
# 회귀에 사용할 데이터가 충분한지 확인
# -----------------------------------
if len(yearly) < 2:
    st.error("회귀분석을 수행할 수 있는 연도가 충분하지 않습니다.")
    st.stop()


# -----------------------------------
# 독립변수 X
# 1908년부터 지난 연수
# -----------------------------------
yearly["지난연수"] = yearly["연도"] - 1908


# -----------------------------------
# 선형회귀
# y = a*x + b
# -----------------------------------
x = yearly["지난연수"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)

slope, intercept = np.polyfit(x, y, 1)


# -----------------------------------
# 상관계수
# -----------------------------------
correlation = np.corrcoef(x, y)[0, 1]


# -----------------------------------
# 회귀식 함수
# -----------------------------------
def predict_temperature(year):
    years_since_1908 = year - 1908
    return slope * years_since_1908 + intercept


# -----------------------------------
# 회귀에 사용된 기간 정보
# -----------------------------------
number_of_years = len(yearly)
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())


# -----------------------------------
# 회귀선용 데이터
# 1900~2100년
# -----------------------------------
prediction_years = np.arange(1900, 2101)

prediction_temperatures = (
    slope * (prediction_years - 1908)
    + intercept
)


# -----------------------------------
# 그래프
# -----------------------------------
fig = go.Figure()


# 실제 연평균기온 산점도
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


# 회귀 직선
fig.add_trace(
    go.Scatter(
        x=prediction_years,
        y=prediction_temperatures,
        mode="lines",
        name="회귀 직선",
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


# 그래프 제목
fig.update_layout(
    title="서울 연평균기온과 회귀 직선",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    xaxis=dict(
        tickmode="linear",
        dtick=10,
        range=[1900, 2100]
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


# -----------------------------------
# 회귀 분석 정보
# -----------------------------------
st.subheader("📊 회귀 분석 정보")

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
        "상관계수",
        f"{correlation:.3f}"
    )


# -----------------------------------
# 회귀식 표시
# -----------------------------------
st.write("### 회귀식")

if intercept >= 0:
    equation_text = (
        f"연평균기온 = "
        f"{slope:.5f} × (연도 - 1908) + "
        f"{intercept:.5f}"
    )
else:
    equation_text = (
        f"연평균기온 = "
        f"{slope:.5f} × (연도 - 1908) "
        f"- {abs(intercept):.5f}"
    )

st.info(equation_text)


# -----------------------------------
# 연도 슬라이더
# -----------------------------------
st.subheader("🔮 기온 예측하기")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)


# -----------------------------------
# 선택한 연도의 예상기온
# -----------------------------------
predicted_temperature = predict_temperature(
    selected_year
)


st.markdown(
    f"""
    <div style="
        text-align: center;
        padding: 25px;
        margin-top: 10px;
        margin-bottom: 20px;
        border-radius: 15px;
        background-color: #f5f5f5;
    ">
        <div style="
            font-size: 24px;
            margin-bottom: 10px;
        ">
            {selected_year}년 예상 연평균기온
        </div>

        <div style="
            font-size: 52px;
            font-weight: bold;
        ">
            {predicted_temperature:.2f} ℃
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------------
# 선택 연도 위치를 그래프에 표시
# -----------------------------------
selected_fig = go.Figure()


selected_fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=7),
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f} ℃"
            "<extra></extra>"
        )
    )
)


selected_fig.add_trace(
    go.Scatter(
        x=prediction_years,
        y=prediction_temperatures,
        mode="lines",
        name="회귀 직선",
        line=dict(width=3),
        hovertemplate=(
            "연도: %{x}년<br>"
            "예상기온: %{y:.2f} ℃"
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


selected_fig.update_layout(
    title=f"{selected_year}년 예상기온 위치",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    xaxis=dict(
        tickmode="linear",
        dtick=10,
        range=[1900, 2100]
    ),
    height=550,
    hovermode="x unified"
)


st.plotly_chart(
    selected_fig,
    width="stretch"
)


# -----------------------------------
# 사용한 데이터 조건
# -----------------------------------
st.subheader("📌 데이터 처리 기준")

st.write(
    "• 2025년까지의 데이터만 사용했습니다."
)

st.write(
    "• 연평균기온을 계산할 때 평균기온이 기록된 관측일만 사용했습니다."
)

st.write(
    "• 관측일수가 300일 미만인 연도는 제외했습니다."
)

st.write(
    "• 회귀분석의 독립변수는 `연도 - 1908`로 설정했습니다."
)

st.write(
    "• 슬라이더의 1900~2100년 예상값은 회귀식을 이용한 값이며, "
    "실제 관측값이 아닙니다."
)
