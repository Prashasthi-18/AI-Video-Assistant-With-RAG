"""Keep users signed in across page refreshes with a session-token cookie."""
import json
import streamlit as st

from core.auth import create_session, user_from_session, delete_session, SESSION_DAYS

COOKIE = "ava_session"


def _write_cookie(value: str, max_age: int):
    # Streamlit can read cookies (st.context.cookies) but not set them, so a tiny script does it.
    st.html(
        f"""<script>
document.cookie = {json.dumps(COOKIE)} + "=" + {json.dumps(value)} + "; path=/; max-age={max_age}; SameSite=Strict"
    + (location.protocol === "https:" ? "; Secure" : "");
</script>""",
        unsafe_allow_javascript=True,
    )


def restore_login():
    """On a fresh page load, sign the user back in from their cookie (if still valid)."""
    if st.session_state.get("user"):
        return
    token = st.context.cookies.get(COOKIE)
    user = user_from_session(token)
    if user:
        st.session_state.user = user
        st.session_state.session_token = token


def login(user: dict, remember: bool = True):
    st.session_state.user = user
    if remember:
        token = create_session(user["id"])
        st.session_state.session_token = token
        st.session_state.pending_cookie = token   # written on the next run, after st.rerun()


def logout():
    delete_session(st.session_state.get("session_token") or st.context.cookies.get(COOKIE))
    st.session_state.clear()
    st.session_state.clear_cookie = True


def flush_cookie_ops():
    """Apply cookie changes queued by login()/logout(). Call once per run."""
    if token := st.session_state.pop("pending_cookie", None):
        _write_cookie(token, SESSION_DAYS * 86400)
    if st.session_state.pop("clear_cookie", False):
        _write_cookie("", 0)
