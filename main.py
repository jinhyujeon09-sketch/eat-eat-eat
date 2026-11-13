import plotly.graph_objects as go
import streamlit as st

from common import (
    APP_TITLE,
    BASE_TIPS,
    LEVEL_COLORS,
    LEVEL_ICONS,
    LEVEL_NAMES,
    LEVEL_TIPS,
    TIPS_SOURCE,
    fetch_today,
    get_final_model,
    level_cuts,
    level_probs,
    level_ranges,
    load_seoul,
)

st.set_page_config(page_title=APP_TITLE, page_icon="🍱", layout="wide")

st.title("🍱 식중독 위험도 평가")
st.write("서울의 기온과 습도로 오늘의 식중독 위험 단계를 알려 주고, 단계에 맞는 예방 활동을 안내해요.")

seoul = load_seoul()
cuts = level_cuts(seoul["발생 건수"])
ranges = level_ranges(cuts)
model = get_final_model()

# ---- 오늘 날씨 ----
mode = st.radio("날씨 넣는 방법", ["자동으로 불러오기", "직접 입력"], horizontal=True)
today = None
if mode == "자동으로 불러오기":
    today = fetch_today()
    if today is None:
        st.warning("날씨를 불러오지 못했어요. 아래 슬라이더로 직접 넣어 주세요.")

if today is not None:
    temp, humid = today["기온"], today["습도"]
    st.caption(f"서울 현재 날씨 (관측 시각 {today['시각']})")
else:
    c1, c2 = st.columns(2)
    temp = c1.slider("기온(℃)", -20.0, 40.0, 25.0, 0.5)
    humid = c2.slider("습도(%)", 0, 100, 70)
    humid = float(humid)

probs = level_probs(model, temp, humid)
level = max(range(4), key=lambda k: probs[k])

m1, m2 = st.columns(2)
m1.metric("기온", f"{temp:.1f}℃")
m2.metric("습도", f"{humid:.0f}%")

# ---- 위험 단계 (글씨 색이 단계에 따라 바뀜) ----
st.subheader("오늘의 식중독 위험 단계")
st.markdown(
    f"<div style='font-size:4rem;font-weight:800;color:{LEVEL_COLORS[level]};line-height:1.2'>"
    f"{LEVEL_ICONS[level]} {LEVEL_NAMES[level]}</div>",
    unsafe_allow_html=True,
)
st.write(
    f"이 날씨는 서울에서 한 달에 식중독이 **{ranges[level]}** 발생하던 때의 날씨와 가장 닮았어요."
)

fig = go.Figure(
    go.Bar(
        x=LEVEL_NAMES,
        y=[p * 100 for p in probs],
        marker_color=LEVEL_COLORS,
        text=[f"{p:.0%}" for p in probs],
        textposition="outside",
    )
)
fig.update_layout(
    title="단계별 확률",
    yaxis_title="확률(%)",
    yaxis_range=[0, 110],
    xaxis_title="위험 단계",
    height=320,
)
st.plotly_chart(fig)

st.caption(
    "이 모델은 서울의 ‘한 달 평균’ 기온·습도로 학습했어요. 오늘의 값을 그 달의 평균처럼 넣어 본 것이라 "
    "참고용이에요. 모델이 얼마나 맞히는지는 ‘모델 성능’ 페이지에서 확인해요."
)

# ---- 예방 활동 ----
st.subheader(f"{LEVEL_ICONS[level]} {LEVEL_NAMES[level]} 단계에서 할 일")
for icon, title, text in LEVEL_TIPS[level]:
    st.checkbox(f"{icon} {title} — {text}", key=f"lvl_{level}_{title}")
with st.expander("언제나 지켜야 할 기본 수칙 6가지", expanded=(level <= 1)):
    for icon, title, text in BASE_TIPS:
        st.checkbox(f"{icon} {title} — {text}", key=f"base_{title}")
st.caption(TIPS_SOURCE)
