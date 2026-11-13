import streamlit as st

from common import (
    APP_TITLE,
    BASE_TIPS,
    LEVEL_COLORS,
    LEVEL_ICONS,
    LEVEL_NAMES,
    LEVEL_TIPS,
    TIPS_SOURCE,
)

st.set_page_config(page_title=APP_TITLE, page_icon="🍱", layout="wide")

st.title("🧼 예방 활동")
st.write("식중독을 막으려면 어떻게 해야 할까요? 기본 수칙과, 위험 단계별로 더 신경 쓸 일을 모았어요.")

st.subheader("기본 수칙 6가지")
for icon, title, text in BASE_TIPS:
    with st.container(border=True):
        st.markdown(f"**{icon} {title}**")
        st.write(text)

st.subheader("위험 단계별로 더 신경 쓸 일")
level = st.radio(
    "단계를 골라 보세요",
    range(4),
    format_func=lambda k: f"{LEVEL_ICONS[k]} {LEVEL_NAMES[k]}",
    horizontal=True,
)
st.markdown(
    f"<div style='font-size:2.2rem;font-weight:800;color:{LEVEL_COLORS[level]}'>"
    f"{LEVEL_ICONS[level]} {LEVEL_NAMES[level]}</div>",
    unsafe_allow_html=True,
)
for icon, title, text in LEVEL_TIPS[level]:
    st.checkbox(f"{icon} {title} — {text}", key=f"page_{level}_{title}")
st.caption(TIPS_SOURCE)
