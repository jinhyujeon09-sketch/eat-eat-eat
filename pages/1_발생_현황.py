import plotly.express as px
import streamlit as st

from common import (
    APP_TITLE,
    LAST_FULL_YEAR,
    LEVEL_COLORS,
    LEVEL_NAMES,
    level_cuts,
    level_ranges,
    load_food,
    load_national,
)

st.set_page_config(page_title=APP_TITLE, page_icon="🍱", layout="wide")

st.title("📊 발생 현황")

food = load_food()
national = load_national()
food_full = food[food["연"] <= LAST_FULL_YEAR]
st.caption(f"연 단위 비교는 2002~{LAST_FULL_YEAR}년이에요. 2026년은 일부 달만 있어서 뺐어요.")

# ---- 1. 월별 평균 ----
st.subheader("몇 월에 많이 발생할까?")
by_month = national.groupby("월", as_index=False)["발생 건수"].mean().round(1)
by_month = by_month.rename(columns={"발생 건수": "월 평균 발생 건수"})
fig = px.bar(
    by_month, x="월", y="월 평균 발생 건수", text="월 평균 발생 건수",
    title="전국 월별 평균 발생 건수 (15개 지역 합계)",
)
fig.update_xaxes(dtick=1)
fig.update_traces(textposition="outside")
st.plotly_chart(fig)
st.caption("서울·경기·인천 등 15개 지역의 합계예요. 세종과 지역을 알 수 없는 ‘불명’은 빠져 있어요.")

# ---- 2. 연도별 추이 ----
st.subheader("해마다 어떻게 달라졌을까?")
c1, c2 = st.columns(2)
areas = ["전국 합계"] + sorted(food_full["지역"].unique())
area = c1.selectbox("지역", areas)
metric = c2.radio("무엇을 볼까요?", ["발생 건수", "환자 수"], horizontal=True)
data = food_full if area == "전국 합계" else food_full[food_full["지역"] == area]
yearly = data.groupby("연", as_index=False)[metric].sum()
fig = px.line(yearly, x="연", y=metric, markers=True, title=f"{area} 연도별 {metric}")
st.plotly_chart(fig)
st.caption("해마다 들쭉날쭉해요. 한 번에 수백 명이 걸리는 집단 발생이 있는 해에 크게 튀어요.")

# ---- 3. 지역별 ----
st.subheader("어느 지역에서 많이 발생했을까?")
c1, c2 = st.columns(2)
metric2 = c1.radio("보는 값", ["발생 건수", "환자 수"], horizontal=True, key="metric2")
years = c2.slider("기간", 2002, LAST_FULL_YEAR, (2015, LAST_FULL_YEAR))
sel = food_full[(food_full["연"] >= years[0]) & (food_full["연"] <= years[1])]
by_area = sel.groupby("지역", as_index=False)[metric2].sum().sort_values(metric2, ascending=False)
fig = px.bar(by_area, x="지역", y=metric2, title=f"지역별 {metric2} 합계 ({years[0]}~{years[1]})")
st.plotly_chart(fig)

# ---- 4. 날씨와의 관계 ----
st.subheader("기온·습도와 발생 건수")
cuts = level_cuts(national["발생 건수"])
ranges = level_ranges(cuts)
st.write(
    "전국 한 달의 발생 건수를 네 단계로 나눠 점 색으로 표시했어요: "
    + ", ".join(f"{n} {r}" for n, r in zip(LEVEL_NAMES, ranges))
    + ". 점 하나가 한 달이에요."
)
color_map = dict(zip(LEVEL_NAMES, LEVEL_COLORS))
c1, c2 = st.columns(2)
with c1:
    fig = px.scatter(
        national, x="평균기온", y="발생 건수", color="단계명",
        color_discrete_map=color_map, category_orders={"단계명": LEVEL_NAMES},
        hover_data=["연월"], title="전국 평균 기온과 월 발생 건수",
        labels={"평균기온": "월 평균 기온(℃)", "단계명": "단계"},
    )
    st.plotly_chart(fig)
with c2:
    fig = px.scatter(
        national, x="평균습도", y="발생 건수", color="단계명",
        color_discrete_map=color_map, category_orders={"단계명": LEVEL_NAMES},
        hover_data=["연월"], title="전국 평균 습도와 월 발생 건수",
        labels={"평균습도": "월 평균 습도(%)", "단계명": "단계"},
    )
    st.plotly_chart(fig)
st.caption("기온·습도가 높을수록 건수가 늘어나는 경향은 있지만, 점이 넓게 퍼져 있어요. 날씨만으로 정확히 맞히기는 어렵다는 뜻이에요.")
