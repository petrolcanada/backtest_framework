"""
Main Streamlit application for visualizing backtest results.
"""
import sys
import streamlit as st
from pathlib import Path

# Add parent directories to path for imports
current_dir = Path(__file__).parent
framework_dir = current_dir.parent.parent
scripts_dir = framework_dir.parent
sys.path.append(str(scripts_dir))

from layouts.dashboard import Dashboard
from utils.loader import load_results
from config import METRIC_CARD_STYLE

# Page configuration
st.set_page_config(
    page_title="Backtest Results Viewer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply custom CSS
st.markdown(METRIC_CARD_STYLE, unsafe_allow_html=True)

# Hide Streamlit default elements for cleaner look
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def main():
    """Main application entry point."""
    # Get results path from command line or use default
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        # Look for the latest results file
        output_dir = framework_dir / "output"
        if output_dir.exists():
            pkl_files = list(output_dir.glob("*.pkl"))
            if pkl_files:
                results_path = str(max(pkl_files, key=lambda p: p.stat().st_mtime))
            else:
                st.error("No results file found. Please run a backtest first.")
                return
        else:
            st.error("Output directory not found. Please run a backtest first.")
            return
    
    try:
        # Load results
        with st.spinner("Loading backtest results..."):
            data, results, engine, strategy_info = load_results(results_path)
        
        # Render dashboard
        dashboard = Dashboard(data, results, engine, strategy_info)
        dashboard.render()
        
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()
