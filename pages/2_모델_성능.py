import pandas as pd
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

from common import (
    APP_TITLE,
    FEATURES,
    LEVEL_NAMES,
    level_cuts,
    level_ranges,
    load_seoul,
    make_logistic,
)

st.set_page_config(page_title=APP_TITLE, page_icon="🍱", layout="wide")

LR_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"
BASE_NAME = "기준 모델(입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델)"

st.title("🧪 모델 성능")
st.write("월 평균 기온과 습도로 그 달의 식중독 위험 단계를 맞히는 모델을 만들고, 얼마나 맞히는지 확인해요.")

seoul = load_seoul()
cuts = level_cuts(seoul["발생 건수"])
ranges = level_ranges(cuts)

# ---- 단계 기준 ----
st.subheader("위험 단계는 이렇게 정했어요")
st.write(
    "서울의 월별 발생 건수를 작은 순서로 줄 세워, 하위 25%·50%·75% 값을 경계로 삼았어요."
)
counts = seoul["단계명"].value_counts().reindex(LEVEL_NAMES, fill_value=0)
st.table(
    pd.DataFrame(
        {"한 달 발생 건수": ranges, "달 수": [f"{n}개월" for n in counts]},
        index=LEVEL_NAMES,
    )
)

# ---- 나누기: 시간 순서 ----
st.subheader("학습용과 테스트용 나누기")
start = st.slider("이 해부터는 테스트용으로 남겨 둬요", 2012, 2023, 2021)
train = seoul[seoul["연"] < start]
test = seoul[seoul["연"] >= start]
st.caption(
    f"학습용 {len(train)}개월 ({train['연'].min()}~{train['연'].max()}년) · "
    f"테스트용 {len(test)}개월 ({test['연'].min()}~{test['연'].max()}년). "
    "미래를 맞히는 모델이라서, 과거로 배우고 이후 해로 시험해요."
)
X_train, y_train = train[FEATURES], train["단계"]
X_test, y_test = test[FEATURES], test["단계"]

lr = make_logistic().fit(X_train, y_train)
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(X_train, y_train)
base = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

scores = {
    LR_NAME: (lr.score(X_train, y_train), lr.score(X_test, y_test)),
    TREE_NAME: (tree.score(X_train, y_train), tree.score(X_test, y_test)),
    BASE_NAME: (base.score(X_train, y_train), base.score(X_test, y_test)),
}

# ---- 정확도 카드 ----
st.subheader("정확도")
for col, (name, (train_acc, test_acc)) in zip(st.columns(3), scores.items()):
    with col:
        with st.container(border=True):
            st.markdown(f"**{name}**")
            st.markdown(
                f"<div style='font-size:2.6rem;font-weight:700;line-height:1.2'>{test_acc:.1%}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"훈련 데이터 {train_acc:.1%}  |  테스트 데이터 {test_acc:.1%}")

lr_test, tree_test, base_test = scores[LR_NAME][1], scores[TREE_NAME][1], scores[BASE_NAME][1]
if tree_test < scores[TREE_NAME][0] - 0.1:
    st.write("의사결정트리는 훈련 때보다 테스트에서 많이 떨어졌어요. 학습용 달을 외우다시피 해서 새 달에는 잘 안 맞는 거예요.")
if lr_test <= base_test:
    st.write("이 구간에서는 로지스틱 회귀가 기준 모델보다 낫지 않아요. 날씨만으로는 한계가 있다는 뜻이에요.")
else:
    st.write(
        f"로지스틱 회귀는 기준 모델보다 {100 * (lr_test - base_test):.1f}%p 더 맞혀요. "
        "날씨가 단서가 되기는 하지만, 단계가 네 개라서 정확히 맞히기는 어려워요."
    )

# ---- 로지스틱 회귀가 어떻게 틀렸나 ----
st.subheader("로지스틱 회귀가 맞힌 모습 (행은 실제, 열은 예측)")
pred = pd.Series(lr.predict(X_test), index=test.index).map(lambda k: LEVEL_NAMES[k])
table = pd.crosstab(test["단계명"], pred).reindex(index=LEVEL_NAMES, columns=LEVEL_NAMES, fill_value=0)
table = table.rename_axis(index="실제", columns="예측")
st.table(table)
st.caption("대각선에 있는 달이 정확히 맞힌 달이에요. 한 단계 차이로 틀린 달이 많은지도 살펴보세요.")

# ---- 해마다 다시 시험해 보기 ----
st.subheader("해마다 다시 시험해 보면")
st.write(
    "위의 점수는 어느 해부터 테스트로 남기느냐에 따라 달라져요. 그래서 2013년부터 2025년까지 "
    "‘그 전 해까지로 배우고 그 해를 맞히기’를 해마다 반복해서 합쳐 보았어요."
)
hits = {"lr": 0, "base": 0}
total = 0
for year in range(2013, 2026):
    tr, te = seoul[seoul["연"] < year], seoul[seoul["연"] == year]
    m_lr = make_logistic().fit(tr[FEATURES], tr["단계"])
    m_base = DummyClassifier(strategy="most_frequent").fit(tr[FEATURES], tr["단계"])
    hits["lr"] += int((m_lr.predict(te[FEATURES]) == te["단계"]).sum())
    hits["base"] += int((m_base.predict(te[FEATURES]) == te["단계"]).sum())
    total += len(te)
r1, r2 = st.columns(2)
r1.metric("로지스틱 회귀(확률로 답하는 모델)", f"{hits['lr'] / total:.1%}")
r2.metric("기준 모델", f"{hits['base'] / total:.1%}")
st.caption(f"{total}개월을 시험한 결과예요. 두 점수가 비슷하면, 서울 기온·습도만으로는 단계를 거의 못 맞힌다는 뜻이에요.")
