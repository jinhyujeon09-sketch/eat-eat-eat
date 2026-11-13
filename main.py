import pandas as pd
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
    get_pred_cuts,
    load_national,
    load_panel,
    predict_count,
    pred_ranges,
    to_level,
)

st.set_page_config(page_title=APP_TITLE, page_icon="🍱", layout="wide")

st.title("🍱 식중독 위험도 평가")
st.write(
    "전국 기온·습도로 이 날씨가 한 달 이어질 때 전국에서 식중독이 얼마나 생길지 예상하고, "
    "위험 단계와 예방 활동을 알려 줘요."
)

national = load_national()
panel = load_panel()
model = get_final_model()
cuts = get_pred_cuts()
ranges = pred_ranges(cuts)
avg_count = float(national["발생 건수"].mean())

# ---- 오늘 날씨 ----
mode = st.radio("날씨 넣는 방법", ["자동으로 불러오기", "직접 입력"], horizontal=True)
today = None
if mode == "자동으로 불러오기":
    today = fetch_today()
    if today is None:
        st.warning("날씨를 불러오지 못했어요. 아래 슬라이더로 직접 넣어 주세요.")

if today is not None:
    temp, humid = today["전국"]["기온"], today["전국"]["습도"]
    st.caption(f"전국 15개 지역 대표 지점의 현재 날씨 평균 (관측 시각 {today['시각']})")
else:
    c1, c2 = st.columns(2)
    temp = c1.slider("전국 평균 기온(℃)", -15.0, 35.0, 20.0, 0.5)
    humid = float(c2.slider("전국 평균 습도(%)", 20, 100, 65))

expected = predict_count(model, temp, humid)
level = to_level(expected, cuts)

m1, m2 = st.columns(2)
m1.metric("전국 평균 기온", f"{temp:.1f}℃")
m2.metric("전국 평균 습도", f"{humid:.0f}%")

# ---- 위험 단계 (글씨 색이 단계에 따라 바뀜) ----
st.subheader("오늘의 식중독 위험 단계")
st.markdown(
    f"<div style='font-size:4rem;font-weight:800;color:{LEVEL_COLORS[level]};line-height:1.2'>"
    f"{LEVEL_ICONS[level]} {LEVEL_NAMES[level]}</div>",
    unsafe_allow_html=True,
)
st.write(
    f"이런 날씨가 한 달 이어지면 전국에서 식중독이 약 **{expected:.0f}건** 생길 것으로 예상돼요. "
    f"(2002년 이후 한 달 평균은 {avg_count:.0f}건이에요.)"
)

fig = go.Figure(
    go.Bar(
        x=LEVEL_NAMES,
        y=[1, 1, 1, 1],
        marker_color=[LEVEL_COLORS[k] if k == level else "#d9d9d9" for k in range(4)],
        text=ranges,
        textposition="inside",
    )
)
fig.update_layout(
    title="위험 단계와 예상 건수의 범위",
    yaxis=dict(visible=False),
    xaxis_title="위험 단계",
    height=240,
    showlegend=False,
)
st.plotly_chart(fig)
st.caption(
    "단계는 과거 모든 달의 ‘예상 건수’를 줄 세워 하위 25%·50%·75%를 경계로 나눈 거예요. "
    "‘모델 성능’ 페이지에서 이 예측이 얼마나 맞는지 확인해요."
)

# ---- 예방 활동 ----
st.subheader(f"{LEVEL_ICONS[level]} {LEVEL_NAMES[level]} 단계에서 할 일")
for icon, title, text in LEVEL_TIPS[level]:
    st.checkbox(f"{icon} {title} — {text}", key=f"lvl_{level}_{title}")
with st.expander("언제나 지켜야 할 기본 수칙 6가지", expanded=(level <= 1)):
    for icon, title, text in BASE_TIPS:
        st.checkbox(f"{icon} {title} — {text}", key=f"base_{title}")
st.caption(TIPS_SOURCE)

# ---- 지역별 참고 ----
with st.expander("지역별 날씨와 발생 비중 (참고)"):
    share = panel.groupby("지역")["발생 건수"].sum()
    share = (share / share.sum() * 100).round(1)
    table = pd.DataFrame({"과거 발생 비중(%)": share})
    if today is not None:
        table["현재 기온(℃)"] = [round(today["지역"][r]["기온"], 1) for r in table.index]
        table["현재 습도(%)"] = [round(today["지역"][r]["습도"]) for r in table.index]
    st.dataframe(table.sort_values("과거 발생 비중(%)", ascending=False))
    st.caption(
        "위험 단계는 전국 기준이에요. 지역 하나만으로는 한 달 발생이 몇 건 안 돼서 날씨로 맞히기 어려웠어요. "
        "지역 표는 참고용이에요."
    )
