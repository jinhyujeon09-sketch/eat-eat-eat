import json
import urllib.request
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent


def find_data_file(patterns, what):
    """data 폴더 또는 앱 폴더(맨 위)에서 이름 패턴에 맞는 파일을 찾는다."""
    for folder in (ROOT / "data", ROOT):
        for pattern in patterns:
            hits = sorted(folder.glob(pattern))
            if hits:
                return hits[0]
    raise FileNotFoundError(
        f"{what} 파일을 찾지 못했어요. 깃허브 맨 위(또는 data 폴더)에 올렸는지 확인해 주세요."
    )


FOOD_PATTERNS = ["식중독*.gz", "*2848*.gz"]
WEATHER_PATTERNS = ["기상_지역별*.csv"]

APP_TITLE = "식중독 위험도 평가"
LEVEL_NAMES = ["안전", "주의", "경고", "위험"]
LEVEL_ICONS = ["🟢", "🟡", "🟠", "🔴"]
LEVEL_COLORS = ["#2e9e5b", "#c99700", "#e8710a", "#d62728"]
FEATURES = ["평균기온", "평균습도"]
LAST_FULL_YEAR = 2025  # 2026년은 일부 달만 있어서 연 단위 비교에서는 뺀다
FIRST_TEST_YEAR = 2013  # 해마다 다시 시험하기를 시작하는 해

# 지역별 날씨는 대표 지점의 값이다 (전남광주는 광주·목포·여수 평균).
# 오늘 날씨를 불러올 때 쓰는 지점 좌표(위도, 경도)
LIVE_POINTS = {
    "서울": [(37.5665, 126.9780)],
    "경기": [(37.2636, 127.0286)],
    "인천": [(37.4563, 126.7052)],
    "강원": [(37.8813, 127.7298)],
    "충북": [(36.6424, 127.4890)],
    "충남": [(36.7845, 126.4503)],
    "대전": [(36.3504, 127.3845)],
    "전북": [(35.8242, 127.1480)],
    "전남광주": [(35.1595, 126.8526), (34.8118, 126.3922), (34.7604, 127.6622)],
    "경북": [(36.5684, 128.7294)],
    "경남": [(35.2279, 128.6811)],
    "대구": [(35.8714, 128.6014)],
    "울산": [(35.5384, 129.3114)],
    "부산": [(35.1796, 129.0756)],
    "제주": [(33.4996, 126.5312)],
}

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
    df = pd.read_csv(
        find_data_file(FOOD_PATTERNS, "식중독 발생 통계"),
        encoding="utf-8-sig",
        dtype={"OCCRNC_MM": str},
    )
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


@st.cache_data
def load_region_weather():
    return pd.read_csv(find_data_file(WEATHER_PATTERNS, "지역별 월별 기상"), encoding="utf-8-sig")


def level_cuts(counts):
    """월 발생 건수의 하위 25%·50%·75% 값. 안전·주의·경고·위험의 경계가 된다."""
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
def load_panel():
    """지역 × 월 표: 지역별 월 발생 건수(없는 달은 0)와 그 지역의 월 평균 기온·습도.
    식중독 자료의 마지막 달은 집계 중일 수 있어 뺀다."""
    food = load_food()
    wx = load_region_weather()
    regions = sorted(wx["지역"].unique())
    last = food["연월"].max()
    months = [m for m in pd.period_range("2002-01", last, freq="M").astype(str) if m < last]
    f = (
        food[food["지역"].isin(regions)]
        .groupby(["지역", "연월"], as_index=False)[["발생 건수", "환자 수"]]
        .sum()
    )
    grid = pd.MultiIndex.from_product([regions, months], names=["지역", "연월"]).to_frame(index=False)
    p = grid.merge(f, on=["지역", "연월"], how="left").fillna({"발생 건수": 0, "환자 수": 0})
    p = p.merge(wx, on=["지역", "연월"], how="left").dropna(subset=FEATURES)
    p["연"] = p["연월"].str[:4].astype(int)
    p["월"] = p["연월"].str[-2:].astype(int)
    return p


@st.cache_data
def load_national():
    """전국(15개 지역 합계) 월별 표: 발생 건수 합계와 지역 평균 기온·습도."""
    p = load_panel()
    n = p.groupby("연월", as_index=False).agg(
        **{
            "발생 건수": ("발생 건수", "sum"),
            "환자 수": ("환자 수", "sum"),
            "평균기온": ("평균기온", "mean"),
            "평균습도": ("평균습도", "mean"),
        }
    )
    n["연"] = n["연월"].str[:4].astype(int)
    n["월"] = n["연월"].str[-2:].astype(int)
    cuts = level_cuts(n["발생 건수"])
    n["단계"] = n["발생 건수"].map(lambda c: to_level(c, cuts))
    n["단계명"] = n["단계"].map(lambda k: LEVEL_NAMES[k])
    return n


# ---------------- 모델 ----------------
def make_count_model():
    """기온·습도로 한 달 발생 건수를 맞히는 포아송 회귀."""
    return make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=2000))


@st.cache_resource
def get_final_model():
    n = load_national()
    return make_count_model().fit(n[FEATURES], n["발생 건수"])


def predict_count(model, temp, humid):
    X = pd.DataFrame([[temp, humid]], columns=FEATURES)
    return float(model.predict(X)[0])


@st.cache_data
def get_pred_cuts():
    """앱이 위험 단계를 정할 때 쓰는 경계: 과거 모든 달의 ‘예상 건수’의 하위 25%·50%·75% 값."""
    n = load_national()
    fitted = pd.Series(get_final_model().predict(n[FEATURES]))
    return level_cuts(fitted)


def pred_ranges(cuts):
    c = [round(x) for x in cuts]
    return [f"{c[0]}건 이하", f"{c[0]}~{c[1]}건", f"{c[1]}~{c[2]}건", f"{c[2]}건 초과"]


@st.cache_data
def rolling_eval():
    """해마다 ‘그 전 해까지로 배우고 그 해를 맞히기’를 반복한 예측표.
    실제 단계는 학습 기간의 실제 건수 경계로, 모델이 답하는 단계는 학습 기간의
    예상 건수 경계로 정한다."""
    n = load_national()
    rows = []
    for year in range(FIRST_TEST_YEAR, int(n["연"].max()) + 1):
        tr, te = n[n["연"] < year], n[n["연"] == year]
        if te.empty:
            continue
        model = make_count_model().fit(tr[FEATURES], tr["발생 건수"])
        by_month = tr.groupby("월")["발생 건수"].mean()
        cuts_act = level_cuts(tr["발생 건수"])
        cuts_w = level_cuts(pd.Series(model.predict(tr[FEATURES])))
        cuts_m = level_cuts(tr["월"].map(by_month))
        major = tr["발생 건수"].map(lambda c: to_level(c, cuts_act)).mode().iloc[0]
        pred_w = model.predict(te[FEATURES])
        for (_, r), pw in zip(te.iterrows(), pred_w):
            pm = float(by_month[r["월"]])
            rows.append(
                {
                    "연월": r["연월"],
                    "연": year,
                    "실제": float(r["발생 건수"]),
                    "날씨 모델": float(pw),
                    "전체 평균": float(tr["발생 건수"].mean()),
                    "달별 평균": pm,
                    "실제 단계": to_level(r["발생 건수"], cuts_act),
                    "날씨 단계": to_level(pw, cuts_w),
                    "달별 단계": to_level(pm, cuts_m),
                    "흔한 단계": int(major),
                }
            )
    return pd.DataFrame(rows)


# ---------------- 오늘 날씨 ----------------
@st.cache_data(ttl=600)
def fetch_today():
    """15개 지역 대표 지점의 현재 기온·습도를 불러와 지역별·전국 평균을 만든다. 실패하면 None."""
    points = [(r, lat, lon) for r, ps in LIVE_POINTS.items() for lat, lon in ps]
    url = (
        "https://api.open-meteo.com/v1/forecast?latitude="
        + ",".join(str(lat) for _, lat, _ in points)
        + "&longitude="
        + ",".join(str(lon) for _, _, lon in points)
        + "&current=temperature_2m,relative_humidity_2m&timezone=Asia%2FSeoul"
    )
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            data = json.loads(r.read().decode("utf-8"))
        if isinstance(data, dict):
            data = [data]
        if len(data) != len(points):
            return None
        per = {}
        for (region, _, _), d in zip(points, data):
            cur = d["current"]
            per.setdefault(region, []).append((float(cur["temperature_2m"]), float(cur["relative_humidity_2m"])))
        regions = {
            region: {
                "기온": sum(v[0] for v in vals) / len(vals),
                "습도": sum(v[1] for v in vals) / len(vals),
            }
            for region, vals in per.items()
        }
        national = {
            "기온": sum(v["기온"] for v in regions.values()) / len(regions),
            "습도": sum(v["습도"] for v in regions.values()) / len(regions),
        }
        return {"전국": national, "지역": regions, "시각": str(data[0]["current"].get("time", ""))}
    except Exception:
        return None
