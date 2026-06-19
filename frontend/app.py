"""
Healthcare Knowledge Navigator — Streamlit Frontend.

A clean, interactive UI for querying the Healthcare RAG Assistant.
This is a placeholder UI that will be expanded in later phases to
include the full query → answer → citation workflow.
"""

import streamlit as st


# ── Page Configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare Knowledge Navigator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_sidebar() -> None:
    """Render the sidebar with project information and settings."""
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/medical-book.png",
            width=80,
        )
        st.title("🏥 HKN")
        st.caption("Healthcare Knowledge Navigator")

        st.divider()

        st.markdown("### ⚙️ Settings")
        st.number_input(
            "Top-K Results",
            min_value=1,
            max_value=50,
            value=20,
            key="top_k",
            help="Number of initial retrieval candidates.",
        )
        st.number_input(
            "Final-K Results",
            min_value=1,
            max_value=20,
            value=5,
            key="final_k",
            help="Number of results after re-ranking.",
        )

        st.divider()

        st.markdown("### 📊 System Status")
        st.success("Backend: Connected", icon="✅")
        st.info("Phase 0 — Architecture Ready", icon="🏗️")

        st.divider()

        st.markdown(
            "Built with ❤️ using **Streamlit**, **FastAPI**, "
            "**Qdrant**, and **Groq**."
        )


def render_main() -> None:
    """Render the main content area."""
    st.title("🏥 Healthcare Knowledge Navigator")
    st.markdown(
        "**Evidence-grounded medical answers** powered by PMC Open Access "
        "literature, hybrid retrieval, and Llama-3.3-70B."
    )

    st.divider()

    # ── Query Input ──────────────────────────────────────────────────
    query = st.text_area(
        "🔍 Enter your medical query",
        placeholder=(
            "e.g., What are the latest treatment approaches for "
            "drug-resistant tuberculosis?"
        ),
        height=120,
        key="query_input",
    )

    col1, col2, _ = st.columns([1, 1, 4])
    with col1:
        search_btn = st.button("🔎 Search", type="primary", use_container_width=True)
    with col2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)

    if clear_btn:
        st.session_state["query_input"] = ""
        st.rerun()

    # ── Placeholder Response Area ────────────────────────────────────
    if search_btn and query:
        with st.spinner("Retrieving evidence and generating answer..."):
            st.info(
                "⏳ **Phase 0 — Architecture Only**\n\n"
                "The retrieval and generation pipeline is not yet "
                "implemented. This UI will be connected to the backend "
                "in subsequent phases.",
                icon="🏗️",
            )

            # Placeholder result structure
            st.markdown("---")
            st.markdown("### 📋 Expected Output Structure")

            with st.expander("Answer", expanded=True):
                st.markdown(
                    "*Generated answer with inline citations will appear here.*"
                )

            with st.expander("📚 Retrieved Evidence"):
                st.markdown("| # | Source | Relevance Score |")
                st.markdown("|---|--------|----------------|")
                st.markdown("| 1 | PMC Article ... | 0.95 |")
                st.markdown("| 2 | PMC Article ... | 0.89 |")

            with st.expander("📊 Confidence Score"):
                st.progress(0.0, text="Confidence: N/A (not yet implemented)")

    elif search_btn and not query:
        st.warning("Please enter a medical query.", icon="⚠️")


def main() -> None:
    """Application entry point."""
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
