
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (
    APP_TITLE,
    BASE_TIPS,
    FEATURES,
    LEVEL_COLORS,
    LEVEL_ICONS,
    LEVEL_NAMES,
    LEVEL_TIPS,
    TIPS_SOURCE,
    fetch_today,
    get_final_model,
    get_pred_cuts,
    get_region_model,
    load_national,
    load_panel,
    predict_count,
    predict_region,
    pred_ranges,
    to_level,
)

st.set_page_config(page_title=APP_TITLE, page_icon="🍱", layout="wide")

st.title("🍱 식중독 위험도 평가")
st.write(
    "기온·습도로 이 날씨가 한 달 이어질 때 식중독이 얼마나 생길지 예상하고, "
    "위험 단계와 예방 활동을 알려 줘요."
)

national = load_national()
panel = load_panel()
nat_model = get_final_model()
reg_model, regions, reg_cuts, reg_fmax, reg_mean = get_region_model()

# ---- 지역 고르기 ----
region = st.selectbox("지역", ["전국"] + regions)

# ---- 오늘 날씨 ----
mode = st.radio("날씨 넣는 방법", ["자동으로 불러오기", "직접 입력"], horizontal=True)
today = None
fetch_error = None
if mode == "자동으로 불러오기":
    try:
        today = fetch_today()
    except Exception as e:
        fetch_error = str(e)
    if today is None:
        st.warning("날씨를 불러오지 못했어요. 아래 슬라이더로 직접 넣어 주세요.")
        with st.expander("실패한 이유 보기"):
            st.code(fetch_error or "알 수 없는 오류")

scope = today.get("범위", "전국") if today else None
live = None
if today is not None:
    if region == "전국":
        live = today["전국"]
        if scope == "전국":
            st.caption(f"전국 15개 지역 대표 지점의 현재 날씨 평균 (관측 시각 {today['시각']})")
        else:
            st.caption(
                f"15개 지역 요청이 실패해서 서울 한 곳의 현재 날씨로 대신했어요 (관측 시각 {today['시각']}). "
                "전국 평균과 다를 수 있어요."
            )
    elif region in today.get("지역", {}):
        live = today["지역"][region]
        st.caption(f"{region} 대표 지점의 현재 날씨 (관측 시각 {today['시각']})")
    else:
        st.warning(f"{region}의 현재 날씨를 받지 못했어요. 아래 슬라이더로 직접 넣어 주세요.")

label = "전국 평균" if region == "전국" else region
if live is not None:
    temp, humid = live["기온"], live["습도"]
else:
    c1, c2 = st.columns(2)
    temp = c1.slider(f"{label} 기온(℃)", -15.0, 35.0, 20.0, 0.5)
    humid = float(c2.slider(f"{label} 습도(%)", 20, 100, 65))

# ---- 예상 ----
if region == "전국":
    expected = predict_count(nat_model, temp, humid)
    cuts = get_pred_cuts()
    fitted_max = float(pd.Series(nat_model.predict(national[FEATURES])).max())
    avg_count = float(national["발생 건수"].mean())
else:
    expected = predict_region(reg_model, regions, region, temp, humid)
    cuts = reg_cuts[region]
    fitted_max = reg_fmax[region]
    avg_count = reg_mean[region]
digits = 0 if region == "전국" else 1
ranges = pred_ranges(cuts, digits)
level = to_level(expected, cuts)

m1, m2 = st.columns(2)
m1.metric(f"{label} 기온", f"{temp:.1f}℃")
m2.metric(f"{label} 습도", f"{humid:.0f}%")

# ---- 위험 단계 (글씨 색이 단계에 따라 바뀜) ----
st.subheader(f"{region}의 오늘 식중독 위험 단계")
st.markdown(
    f"<div style='font-size:4rem;font-weight:800;color:{LEVEL_COLORS[level]};line-height:1.2'>"
    f"{LEVEL_ICONS[level]} {LEVEL_NAMES[level]}</div>",
    unsafe_allow_html=True,
)
st.write(
    f"이런 날씨가 한 달 이어지면 {region}에서 식중독이 약 **{expected:.{digits}f}건** 생길 것으로 예상돼요. "
    f"(2002년 이후 한 달 평균은 {avg_count:.{digits}f}건이에요.)"
)
if region != "전국":
    st.info(
        "지역 하나는 한 달 발생이 몇 건뿐이라 날씨로 맞히기가 더 어려워요. "
        f"단계는 {region}의 과거 기록과 비교한 순위이니 참고용으로 봐 주세요."
    )

# 막대 높이 = 그 단계의 예상 건수 상한 (위험은 과거 예상 건수의 최댓값)
heights = [cuts[0], cuts[1], cuts[2], max(fitted_max, expected)]
fig = go.Figure(
    go.Bar(
        x=LEVEL_NAMES,
        y=heights,
        marker_color=[LEVEL_COLORS[k] if k == level else "#d9d9d9" for k in range(4)],
        text=ranges,
        textposition="inside",
    )
)
fig.add_hline(
    y=expected,
    line_dash="dash",
    line_color=LEVEL_COLORS[level],
    annotation_text=f"오늘 예상 {expected:.{digits}f}건",
    annotation_position="top left",
)
fig.update_layout(
    title="위험 단계와 예상 건수의 범위",
    yaxis_title="한 달 예상 발생 건수(건)",
    yaxis_range=[0, max(heights) * 1.2],
    xaxis_title="위험 단계",
    height=360,
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
        table["현재 기온(℃)"] = [
            round(today["지역"][r]["기온"], 1) if r in today.get("지역", {}) else None for r in table.index
        ]
        table["현재 습도(%)"] = [
            round(today["지역"][r]["습도"]) if r in today.get("지역", {}) else None for r in table.index
        ]
    st.dataframe(table.sort_values("과거 발생 비중(%)", ascending=False))
    st.caption("세종과 지역을 알 수 없는 ‘불명’은 자료가 적어 빠져 있어요.")
