import streamlit as st

from core.auth import sign_in, sign_up, AuthError
from ui.icons import icon, LOGO_SVG
from ui.session import login

TRUST = [("zap", "Captions in seconds"), ("languages", "English & Hinglish"),
         ("shield", "Private by default"), ("database", "RAG-grounded answers")]

WAVE_BARS = "".join("<span></span>" for _ in range(22))


def _preview_html() -> str:
    """A tiny animated mock of the app: a video, its summary, and a chat answer appearing."""
    return f"""
<div class="preview">
  <div class="pv-top"><div class="pv-dots"><i></i><i></i><i></i></div>
    <div class="pv-url">youtube.com/watch?v=UF8uR6Z6KLc</div></div>
  <div class="pv-body">
    <div class="pv-video"><div class="pv-play">{icon("play", 16)}</div><div class="pv-wave">{WAVE_BARS}</div></div>
    <div class="pv-side">
      <div class="pv-label">Summary</div>
      <div class="pv-line" style="width:95%"></div><div class="pv-line" style="width:80%"></div>
      <div class="pv-line" style="width:88%"></div>
      <div class="pv-msg q">What's the key takeaway?</div>
      <div class="pv-msg a"><span class="typing"><i></i><i></i><i></i></span>
        <span class="ans">Trust that the dots will connect — follow what you love.</span></div>
    </div>
  </div>
  <div class="pv-float f1">{icon("circle-check", 15)} 15-min talk summarised in 6s</div>
  <div class="pv-float f2">{icon("sparkles", 15)} 3 key decisions found</div>
</div>"""


def render_auth_page():
    # Auth screen: no sidebar, no header bar, no scrolling — one full-height, vertically centred view.
    st.markdown(
        """<style>
section[data-testid='stSidebar'], [data-testid='stSidebarCollapsedControl'],
[data-testid='stExpandSidebarButton'], header[data-testid='stHeader'] { display: none; }
[data-testid='stMain'] { overflow: hidden; }
[data-testid='stMainBlockContainer'] {
    height: 100vh; padding-top: 0 !important; padding-bottom: 0 !important;
    display: flex; flex-direction: column; justify-content: center;
}
[data-testid='stMainBlockContainer'] > [data-testid='stVerticalBlock'] { justify-content: center; }
</style>""",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15, 1], gap="large", vertical_alignment="center")

    with left:
        trust = "".join(f"<span>{icon(ic, 13)}{label}</span>" for ic, label in TRUST)
        st.markdown(
            f"""
<div class="auth-brand">
  <div class="brand big">{LOGO_SVG}<div><div class="brand-name">AI Video Assistant</div>
    <div class="brand-tag">Transcribe · Summarise · Chat</div></div></div>
  <h1>Your videos,<br><span class="grad-anim">finally searchable.</span></h1>
  <p class="sub">Turn lectures, meetings and YouTube videos into summaries you can read in a minute
  — and a chat you can ask anything.</p>
  {_preview_html()}
  <div class="trust">{trust}</div>
</div>""",
            unsafe_allow_html=True,
        )

    with right:
        with st.container(key="auth_card"):
            tab_in, tab_up = st.tabs([":material/login: Sign in", ":material/person_add: Create account"])

            with tab_in:
                st.markdown('<h3 class="auth-title">Welcome back</h3>'
                            '<p class="auth-sub">Sign in to continue to your workspace.</p>',
                            unsafe_allow_html=True)
                with st.form("signin_form", border=False):
                    email = st.text_input("Email", placeholder="you@example.com", key="in_email")
                    password = st.text_input("Password", type="password", placeholder="••••••••", key="in_password")
                    remember = st.checkbox("Keep me signed in for 30 days", value=True)
                    submitted = st.form_submit_button("Sign in", type="primary",
                                                      icon=":material/login:", use_container_width=True)
                if submitted:
                    try:
                        login(sign_in(email, password), remember)
                        st.rerun()
                    except AuthError as e:
                        st.error(str(e), icon=":material/error:")
                st.markdown('<p class="auth-foot">New here? Open the <b>Create account</b> tab above.</p>',
                            unsafe_allow_html=True)

            with tab_up:
                st.markdown('<h3 class="auth-title">Create your account</h3>'
                            '<p class="auth-sub">Free forever. Takes 20 seconds.</p>',
                            unsafe_allow_html=True)
                with st.form("signup_form", border=False):
                    name = st.text_input("Full name", placeholder="Jane Doe")
                    email = st.text_input("Email", placeholder="you@example.com", key="up_email")
                    c1, c2 = st.columns(2)
                    password = c1.text_input("Password", type="password", placeholder="8+ chars, letters & numbers")
                    confirm = c2.text_input("Confirm password", type="password", placeholder="Repeat password")
                    submitted = st.form_submit_button("Create account", type="primary",
                                                      icon=":material/person_add:", use_container_width=True)
                if submitted:
                    try:
                        login(sign_up(name, email, password, confirm))
                        st.session_state.show_welcome = True   # onboarding dialog on first visit
                        st.rerun()
                    except AuthError as e:
                        st.error(str(e), icon=":material/error:")

            st.markdown(f'<div class="auth-secure">{icon("shield", 14)} Passwords are hashed with bcrypt '
                        'and stored locally.</div>', unsafe_allow_html=True)
