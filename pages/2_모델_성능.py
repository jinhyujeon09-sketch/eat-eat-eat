import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (
    APP_TITLE,
    FIRST_TEST_YEAR,
    LEVEL_NAMES,
    level_cuts,
    level_ranges,
    load_national,
    rolling_eval,
)

st.set_page_config(page_title=APP_TITLE, page_icon="🍱", layout="wide")

PO_NAME = "포아송 회귀(건수를 맞히는 모델)"
MONTH_NAME = "달별 평균 모델(그 달의 과거 평균으로 답하는 모델)"
BASE_NAME = "기준 모델(과거 전체 평균으로만 답하는 모델)"

st.title("🧪 모델 성능")
st.write(
    "전국 평균 기온·습도로 한 달 발생 건수를 맞히는 모델을 만들고, 얼마나 맞히는지 확인해요. "
    "미래를 맞히는 모델이라서, 해마다 ‘그 전 해까지로 배우고 그 해를 시험’해요."
)

national = load_national()
res = rolling_eval()
n_months = len(res)
st.caption(
    f"{FIRST_TEST_YEAR}년부터 {int(res['연'].max())}년까지 {n_months}개월을 시험했어요. "
    "한 번 나누기보다 운에 덜 휘둘려요."
)

# ---- 1. 건수를 얼마나 맞혔나 ----
st.subheader("발생 건수를 맞힌 정도")
mae = {
    BASE_NAME: float((res["전체 평균"] - res["실제"]).abs().mean()),
    MONTH_NAME: float((res["달별 평균"] - res["실제"]).abs().mean()),
    PO_NAME: float((res["날씨 모델"] - res["실제"]).abs().mean()),
}
for col, (name, value) in zip(st.columns(3), mae.items()):
    with col:
        with st.container(border=True):
            st.markdown(f"**{name}**")
            st.markdown(
                f"<div style='font-size:2.6rem;font-weight:700;line-height:1.2'>{value:.1f}건</div>",
                unsafe_allow_html=True,
            )
            gain = 1 - value / mae[BASE_NAME]
            st.caption("평균 오차" if name == BASE_NAME else f"평균 오차 · 기준보다 {gain:.0%} 작음")
st.write(
    "평균 오차는 ‘예상한 건수와 실제 건수가 평균 몇 건 차이 났나’예요. 작을수록 좋아요. "
    f"날씨 모델({mae[PO_NAME]:.1f}건)과 달별 평균 모델({mae[MONTH_NAME]:.1f}건)이 비슷하면, "
    "기온·습도가 계절 말고 더 알려 주는 게 거의 없다는 뜻이에요."
)

# ---- 2. 단계를 얼마나 맞혔나 ----
st.subheader("위험 단계를 맞힌 정도")
cuts = level_cuts(national["발생 건수"])
st.write(
    "‘정답 단계’는 그 시점까지의 실제 건수를 줄 세워 하위 25%·50%·75%로 나눈 단계예요. "
    "전체 기간으로 나누면 "
    + ", ".join(f"{n} {r}" for n, r in zip(LEVEL_NAMES, level_ranges(cuts)))
    + "예요."
)
hits = {
    "기준 모델(가장 흔한 단계로만 답하는 모델)": float((res["흔한 단계"] == res["실제 단계"]).mean()),
    MONTH_NAME: float((res["달별 단계"] == res["실제 단계"]).mean()),
    PO_NAME: float((res["날씨 단계"] == res["실제 단계"]).mean()),
}
for col, (name, value) in zip(st.columns(3), hits.items()):
    with col:
        with st.container(border=True):
            st.markdown(f"**{name}**")
            st.markdown(
                f"<div style='font-size:2.6rem;font-weight:700;line-height:1.2'>{value:.1%}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"{n_months}개월 중 {round(value * n_months)}개월을 맞힘")
st.caption("단계가 4개라 아무렇게나 찍어도 25% 안팎은 맞아요. 그보다 얼마나 높은지를 보세요.")

# ---- 3. 예상과 실제 ----
st.subheader("예상과 실제 비교")
fig = go.Figure()
fig.add_trace(go.Scatter(x=res["연월"], y=res["실제"], mode="lines", name="실제 건수", line=dict(width=2)))
fig.add_trace(go.Scatter(x=res["연월"], y=res["날씨 모델"], mode="lines", name="날씨 모델 예상", line=dict(width=2, dash="dot")))
fig.update_layout(xaxis_title="연월", yaxis_title="전국 월 발생 건수", height=420)
st.plotly_chart(fig)

# ---- 4. 어떻게 틀렸나 ----
st.subheader("날씨 모델이 단계를 맞힌 모습 (행은 정답, 열은 예측)")
table = pd.crosstab(
    res["실제 단계"].map(lambda k: LEVEL_NAMES[k]),
    res["날씨 단계"].map(lambda k: LEVEL_NAMES[k]),
).reindex(index=LEVEL_NAMES, columns=LEVEL_NAMES, fill_value=0)
st.table(table.rename_axis(index="정답", columns="예측"))
st.caption("대각선에 있는 달이 정확히 맞힌 달이에요. 한 단계 차이로 틀린 달이 많은지도 살펴보세요.")
