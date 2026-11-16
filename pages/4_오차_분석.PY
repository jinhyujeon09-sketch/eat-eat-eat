import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import APP_TITLE, FIRST_TEST_YEAR, load_national, load_panel, rolling_eval

st.set_page_config(page_title=APP_TITLE, page_icon="🍱", layout="wide")

st.title("🔎 오차 분석")
st.write(
    "모델이 예상한 건수와 실제 건수가 크게 어긋난 달을 찾아보고, 왜 그랬을지 단서를 모아 봐요. "
    "아래 ‘원인 후보’는 데이터로 증명한 것이 아니라 기사와 데이터에서 짐작한 것이에요."
)

# ---------------- 데이터 ----------------
nat = load_national()
panel = load_panel()
res = rolling_eval().copy()
res["오차"] = res["실제"] - res["날씨 모델"]  # +면 예상보다 많이 발생
res["절대 오차"] = res["오차"].abs()
climate = nat.groupby("월")["평균기온"].mean()
res = res.merge(nat[["연월", "월", "평균기온", "환자 수"]], on="연월")
res["기온 편차"] = res["평균기온"] - res["월"].map(climate)  # 그 달 평년(2002년 이후 평균) 대비

st.caption(
    f"{FIRST_TEST_YEAR}년부터 {res['연월'].max()}까지 {len(res)}개월을 ‘그 전 해까지로 배우고 그 해를 맞히기’로 시험한 결과예요. "
    "오차 = 실제 − 예상이고, 양수이면 예상보다 많이 발생한 거예요."
)

# ---------------- 1. 요약 ----------------
st.subheader("얼마나 틀렸나")
c1, c2, c3 = st.columns(3)
for col, label, value in [
    (c1, "중앙값 오차", res["절대 오차"].median()),
    (c2, "평균 오차", res["절대 오차"].mean()),
    (c3, "큰 오차의 기준(상위 10%)", res["절대 오차"].quantile(0.9)),
]:
    with col:
        with st.container(border=True):
            st.metric(label, f"{value:.1f}건")

# ---------------- 2. 연도별 쏠림 ----------------
st.subheader("해마다 예상보다 많았나, 적었나")
by_year = res.groupby("연", as_index=False)["오차"].mean()
fig = go.Figure(
    go.Bar(
        x=by_year["연"],
        y=by_year["오차"],
        marker_color=["#d62728" if v > 0 else "#4c78a8" for v in by_year["오차"]],
        text=[f"{v:+.1f}" for v in by_year["오차"]],
        textposition="outside",
    )
)
fig.update_layout(xaxis_title="연도", yaxis_title="평균 오차(건/월)", height=360)
fig.update_xaxes(dtick=1)
st.plotly_chart(fig)
y26 = by_year[by_year["연"] == by_year["연"].max()]
if not y26.empty:
    st.write(
        f"가장 최근 해({int(y26['연'].iloc[0])}년)는 한 달에 평균 {float(y26['오차'].iloc[0]):+.1f}건 어긋났어요. "
        "빨간 막대는 예상보다 많이 발생한 해, 파란 막대는 적게 발생한 해예요."
    )

# ---------------- 3. 가장 크게 틀린 달 ----------------
st.subheader("가장 크게 틀린 달")
top_n = st.slider("몇 개월을 볼까요?", 5, 20, 10)
top = res.sort_values("절대 오차", ascending=False).head(top_n)
table = top[["연월", "실제", "날씨 모델", "오차", "환자 수", "기온 편차"]].rename(
    columns={"날씨 모델": "예상", "기온 편차": "기온 편차(평년 대비, ℃)"}
)
st.dataframe(table.round(1), hide_index=True)

# ---------------- 4. 날씨 때문일까? ----------------
st.subheader("날씨가 평소와 달라서 틀린 걸까?")
res["기온 편차 크기"] = res["기온 편차"].abs()
corr = res["절대 오차"].corr(res["기온 편차 크기"])
fig = px.scatter(res, x="기온 편차 크기", y="절대 오차", hover_data=["연월"])
fig.update_layout(
    xaxis_title="그 달 기온이 평년과 다른 정도(℃)",
    yaxis_title="예상과 실제의 차이(건)",
    height=360,
)
st.plotly_chart(fig)
st.write(
    f"두 값의 상관은 {corr:.2f}예요. 0에 가까우면 ‘날씨가 평소와 많이 달라서 크게 틀린 것’은 아니라는 뜻이에요. "
    "큰 오차의 이유는 날씨 밖에 있다는 뜻이기도 해요."
)

# ---------------- 5. 한 달 자세히 보기 ----------------
st.subheader("한 달 자세히 보기")
ym = st.selectbox("달을 골라 보세요", list(top["연월"]))
row = res[res["연월"] == ym].iloc[0]
mo = int(ym[-2:])
year = int(ym[:4])
st.write(
    f"**{ym}**: 실제 {row['실제']:.0f}건, 예상 {row['날씨 모델']:.0f}건 (오차 {row['오차']:+.0f}건), "
    f"환자 {row['환자 수']:.0f}명, 기온은 평년보다 {row['기온 편차']:+.1f}℃."
)
mon_mean = panel.groupby(["지역", "월"])["발생 건수"].mean()
sel = panel[panel["연월"] == ym].set_index("지역")["발생 건수"]
contrib = pd.Series({r: float(sel[r] - mon_mean[(r, mo)]) for r in sel.index}).sort_values()
fig = go.Figure(
    go.Bar(
        x=contrib.values,
        y=contrib.index,
        orientation="h",
        marker_color=["#d62728" if v > 0 else "#4c78a8" for v in contrib.values],
    )
)
fig.update_layout(
    title=f"{ym}에 평소 같은 달보다 지역별로 얼마나 많았나",
    xaxis_title="평소 같은 달 평균과의 차이(건)",
    height=460,
)
st.plotly_chart(fig)
st.caption("한두 지역만 튀었다면 지역 사고, 여러 큰 지역이 함께 올랐다면 전국적인 변화일 가능성이 커요.")

# ---------------- 6. 정리해 둔 기사 ----------------
st.subheader("참고 기사")
st.caption("제가 찾아서 정리한 기사예요. 제목과 링크, 짧은 요약만 적었어요. 기사가 곧 원인이라는 뜻은 아니에요.")
ARTICLES = [
    {
        "연": 2026,
        "제목": "경기도 식중독 비상…점검 및 예방 홍보 강화를 (사설)",
        "매체": "경기신문",
        "날짜": "2026-08-06",
        "링크": "https://kgnews.co.kr/news/article.html?no=906592",
        "요약": "올해 상반기 경기도 발생 건수가 작년 한 해의 약 1.8배라는 내용이에요. 식약처는 때 이른 더위와 큰 일교차를 원인으로 본다고 해요. 5월 용인의 대형 매장 집단 의심 사고도 다뤄요.",
    },
    {
        "연": 2026,
        "제목": "노로바이러스 급증…추석 식중독 주의보",
        "매체": "세계일보",
        "날짜": "2026-09-24",
        "링크": "https://www.segye.com/newsView/20260924509283",
        "요약": "겨울에 주로 도는 노로바이러스가 올여름 서울에서 이례적으로 많이 검출됐다는 서울시 보건환경연구원 발표를 전해요.",
    },
    {
        "연": 2026,
        "제목": "하루 한 번 식중독 공포…AI가 ‘식탁 안전’ 지킬까?",
        "매체": "의약일보",
        "날짜": "2026-05",
        "링크": "https://www.medicaldaily.co.kr/news/19304",
        "요약": "최근 3년 집단식중독 현황과, 식약처가 올해 AI 기반 원인 추정 시스템을 도입했다는 소식, 5월 서울 냉면 전문점 사례를 다뤄요.",
    },
    {
        "연": 2026,
        "제목": "경남 식중독 3년째 증가세",
        "매체": "경남신문",
        "날짜": "2026-11-04",
        "링크": "http://www.knnews.co.kr/news/articleView.php?idxno=1554408",
        "요약": "경남의 환자 수가 3년째 늘었고, 전국적으로 겨울철 환자가 전년보다 두 배 이상 늘었다는 내용이에요.",
    },
    {
        "연": 2026,
        "제목": "덥고 습한 날씨에 식중독 급증…도시락 섭취 주의",
        "매체": "코메디닷컴",
        "날짜": "2026-07-30",
        "링크": "https://kormedi.com/2727546/",
        "요약": "도시락 식중독이 해마다 늘고, 올해는 6월까지 12건으로 예년보다 빠르다는 식약처 자료를 전해요.",
    },
    {
        "연": 2025,
        "제목": "‘여름 지났다고 방심했더니’…가을철 식중독, 9월에 가장 많았다",
        "매체": "세계일보",
        "날짜": "2026-10-10",
        "링크": "https://www.segye.com/newsView/20261010503607",
        "요약": "2025년에는 9월 식중독이 연중 가장 많았고, 겨울철 발생도 전년보다 크게 늘었다는 내용이에요.",
    },
]
rel = [a for a in ARTICLES if a["연"] == year]
others = [a for a in ARTICLES if a["연"] != year]


def show_article(a):
    with st.container(border=True):
        st.markdown(f"**[{a['제목']}]({a['링크']})**")
        st.caption(f"{a['매체']} · {a['날짜']}")
        st.write(a["요약"])


if rel:
    st.markdown(f"**{year}년과 관련된 기사**")
    for a in rel:
        show_article(a)
if others:
    st.markdown("**그 밖의 기사**" if rel else "**기사 목록**")
    for a in others:
        show_article(a)

# ---------------- 7. 원인 후보 ----------------
st.subheader("원인 후보 (확인이 더 필요해요)")
st.table(
    pd.DataFrame(
        [
            ["최근 해에 예상보다 훨씬 많음", "여러 큰 지역이 함께 늘었고, 기온은 평년과 비슷했어요", "집단 발생 사고, 노로바이러스의 이례적 유행, 신고·집계 방식의 변화", "추정"],
            ["겨울에 예상보다 많은 달", "추운 달인데 실제가 예상의 서너 배인 달이 있어요", "겨울에 도는 노로바이러스 (모델은 ‘따뜻할수록 많다’만 배웠어요)", "추정"],
            ["2020년에 예상보다 적음", "2~5월 건수가 몇 건으로 줄었어요", "코로나 시기의 모임·급식 변화", "추정"],
            ["건수에 비해 환자가 매우 많은 달", "환자 수가 건수에 비해 아주 큰 달이 있어요", "학교 급식 같은 대규모 집단 발생", "추정"],
        ],
        columns=["현상", "데이터가 보여 주는 것", "원인 후보", "상태"],
    )
)

# ---------------- 8. 최근 기사 찾아보기 ----------------
st.subheader("최근 기사 찾아보기")
st.caption("선택한 달로 뉴스 검색 결과를 불러와요. 제목과 링크만 보여 주고, 검색 결과가 이 달의 오차와 관련 있는지는 직접 판단해야 해요.")


def parse_rss(xml_text, limit=8):
    root = ET.fromstring(xml_text)
    items = []
    for it in root.iter("item"):
        src = it.find("source")
        items.append(
            {
                "제목": (it.findtext("title") or "").strip(),
                "링크": (it.findtext("link") or "").strip(),
                "날짜": (it.findtext("pubDate") or "").strip(),
                "매체": (src.text or "").strip() if src is not None and src.text else "",
            }
        )
        if len(items) >= limit:
            break
    return items


def fetch_news(query):
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": query, "hl": "ko", "gl": "KR", "ceid": "KR:ko"}
    )
    req = urllib.request.Request(url, headers={"User-Agent": "food-risk-app/1.0"})
    with urllib.request.urlopen(req, timeout=8) as r:
        return parse_rss(r.read().decode("utf-8"))


query = f"식중독 {year}년 {mo}월"
st.write(f"검색어: **{query}**")
if st.button("기사 찾아보기"):
    try:
        found = fetch_news(query)
    except Exception as e:
        found = None
        st.info("지금은 기사를 불러오지 못했어요. 위에 정리해 둔 기사를 참고해 주세요.")
        with st.expander("실패한 이유 보기"):
            st.code(f"{type(e).__name__}: {e}")
    if found is not None:
        if not found:
            st.write("검색 결과가 없어요.")
        for a in found:
            st.markdown(f"- [{a['제목']}]({a['링크']})  \n  {a['매체']} · {a['날짜']}")
        st.caption("자동 검색 결과예요. 이 달의 오차와 관련 있는지 직접 확인해 주세요.")
