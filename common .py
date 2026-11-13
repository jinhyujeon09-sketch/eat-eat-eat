import json
import urllib.request
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).parent / "data"
FOOD_FILE = DATA_DIR / "식중독_지역월별.csv.gz"
WEATHER_FILE = DATA_DIR / "서울_기상_월별.csv"

APP_TITLE = "식중독 위험도 평가"
LEVEL_NAMES = ["안전", "주의", "경고", "위험"]
LEVEL_ICONS = ["🟢", "🟡", "🟠", "🔴"]
LEVEL_COLORS = ["#2e9e5b", "#c99700", "#e8710a", "#d62728"]
FEATURES = ["평균기온", "평균습도"]
LAST_FULL_YEAR = 2025  # 2026년은 9월까지만 있어서 연 단위 비교에서는 뺀다

# ---- 식품의약품안전처 식중독 예방 6대 수칙 (숫자는 식약처 안내 기준) ----
BASE_TIPS = [
    ("🧼", "손 씻기", "흐르는 물에 비누로 30초 이상 씻어요."),
    ("🔥", "익혀 먹기", "육류·가금류는 중심온도 75℃에서 1분 이상, 어패류는 85℃에서 1분 이상 가열해요."),
    ("❄️", "보관 온도 지키기", "냉장 식품은 5℃ 이하, 냉동 식품은 -18℃ 이하로 보관해요."),
    ("🔪", "구분 사용하기", "칼·도마는 채소용, 육류용, 어류용으로 나눠 써요."),
    ("🚿", "세척·소독하기", "식재료는 깨끗이 씻고, 주방기구는 소독해요."),
    ("💧", "끓여 먹기", "물은 끓여서 마셔요."),
]
LEVEL_TIPS = {
    0: [("✅", "지금처럼 기본 수칙 지키기", "날씨가 안전해도 손 씻기와 익혀 먹기는 늘 지켜요.")],
    1: [("🛒", "장 본 뒤 바로 냉장", "육류·달걀은 장보기 마지막에 사서 조리 전까지 냉장고에 넣어요.")],
    2: [
        ("⏱️", "조리한 음식은 2시간 안에", "가열한 음식은 가능한 2시간 안에 먹고, 바로 못 먹으면 식혀서 바로 냉장해요."),
        ("🥬", "채소는 씻은 뒤 바로", "채소는 씻은 뒤 바로 먹거나 냉장 보관해요."),
    ],
    3: [
        ("⏱️", "조리한 음식은 2시간 안에", "가열한 음식은 가능한 2시간 안에 먹고, 바로 못 먹으면 식혀서 바로 냉장해요."),
        ("🧊", "야외에서는 아이스박스", "도시락·김밥을 차 안이나 트렁크에 2시간 넘게 두지 말고, 아이스박스로 10℃ 이하를 지켜요."),
        ("🥬", "채소는 씻은 뒤 바로", "채소는 씻은 뒤 바로 먹거나 냉장 보관해요."),
    ],
}
TIPS_SOURCE = "출처: 식품의약품안전처 식중독 예방 6대 수칙 (최신 내용은 식약처 누리집에서 확인하세요)"


# ---------------- 데이터 ----------------
@st.cache_data
def load_food():
    df = pd.read_csv(FOOD_FILE, encoding="utf-8-sig", dtype={"OCCRNC_MM": str})
    df = df.rename(
        columns={
            "OCCRNC_YEAR": "연",
            "OCCRNC_MM": "월",
            "OCCRNC_AREA": "지역",
            "OCCRNC_CNT": "발생 건수",
            "PATNT_CNT": "환자 수",
        }
    )
    df["월"] = df["월"].astype(int)
    df["연월"] = df["연"].astype(str) + "-" + df["월"].astype(str).str.zfill(2)
    return df


def _pick(df, prefix):
    for c in df.columns:
        if c.startswith(prefix):
            return c
    raise KeyError(prefix)


def level_cuts(counts):
    """발생 건수의 하위 25%·50%·75% 값. 안전·주의·경고·위험의 경계가 된다."""
    return [float(counts.quantile(q)) for q in (0.25, 0.5, 0.75)]


def to_level(count, cuts):
    for k, cut in enumerate(cuts):
        if count <= cut:
            return k
    return 3


def _range_text(lo, hi):
    if hi is None:
        return f"{lo}건 이상"
    return f"{lo}건" if lo == hi else f"{lo}~{hi}건"


def level_ranges(cuts):
    c = [int(x) for x in cuts]
    return [
        _range_text(0, c[0]),
        _range_text(c[0] + 1, c[1]),
        _range_text(c[1] + 1, c[2]),
        _range_text(c[2] + 1, None),
    ]


@st.cache_data
def load_seoul():
    """서울 월별 기온·습도·강수량에 서울 월별 식중독(없는 달은 0건)을 붙인 표."""
    w = pd.read_csv(WEATHER_FILE, encoding="cp949")
    w = w[[_pick(w, "일시"), _pick(w, "평균기온("), _pick(w, "평균상대습도("), _pick(w, "월합강수량(")]]
    w.columns = ["연월", "평균기온", "평균습도", "강수량"]
    food = load_food()
    s = food[food["지역"] == "서울"][["연월", "발생 건수", "환자 수"]]
    m = w.merge(s, on="연월", how="left").fillna({"발생 건수": 0, "환자 수": 0})
    m["연"] = m["연월"].str[:4].astype(int)
    m["월"] = m["연월"].str[-2:].astype(int)
    cuts = level_cuts(m["발생 건수"])
    m["단계"] = m["발생 건수"].map(lambda c: to_level(c, cuts))
    m["단계명"] = m["단계"].map(lambda k: LEVEL_NAMES[k])
    return m


def make_logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


@st.cache_resource
def get_final_model():
    """서울 전체 기간(2005~2025)으로 학습한 최종 모델."""
    m = load_seoul()
    return make_logistic().fit(m[FEATURES], m["단계"])


def level_probs(model, temp, humid):
    X = pd.DataFrame([[temp, humid]], columns=FEATURES)
    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)
    return [float(proba[classes.index(k)]) if k in classes else 0.0 for k in range(4)]


# ---------------- 오늘 날씨 ----------------
@st.cache_data(ttl=600)
def fetch_today():
    """서울의 현재 기온·습도. 실패하면 None."""
    url = (
        "https://api.open-meteo.com/v1/forecast?latitude=37.5665&longitude=126.9780"
        "&current=temperature_2m,relative_humidity_2m&timezone=Asia%2FSeoul"
    )
    try:
        with urllib.request.urlopen(url, timeout=6) as r:
            cur = json.loads(r.read().decode("utf-8"))["current"]
        return {
            "기온": float(cur["temperature_2m"]),
            "습도": float(cur["relative_humidity_2m"]),
            "시각": str(cur.get("time", "")),
        }
    except Exception:
        return None
