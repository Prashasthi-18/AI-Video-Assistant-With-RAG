import streamlit as st

# A soft violet light that follows the cursor (drawn by CSS from --mx / --my).
_SPOTLIGHT_JS = """<script>
if (!window.__avaSpotlight) {
  window.__avaSpotlight = true;
  let raf = 0;
  document.addEventListener("pointermove", (e) => {
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => {
      document.documentElement.style.setProperty("--mx", e.clientX + "px");
      document.documentElement.style.setProperty("--my", e.clientY + "px");
    });
  }, { passive: true });
}
</script>"""


def inject_effects():
    st.html(_SPOTLIGHT_JS, unsafe_allow_javascript=True)
