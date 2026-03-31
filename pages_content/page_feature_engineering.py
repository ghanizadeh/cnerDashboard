"""
pages_content/page_feature_engineering.py
Standalone Feature Engineering tool — accessible via Extra Tool menu.
"""

import pandas as pd
import streamlit as st
from core.data.feature_engineering import (
    FeatureEngineeringConfig,
    apply_feature_engineering,
)
from state.session import init_state, get_value, set_state


def render():
    init_state()

    st.title("🔧 Feature Engineering")
    st.markdown(
        """
        This tool allows you to apply various feature engineering techniques to your dataset.
        Select the desired transformations and click "Apply" to see the results.
        """
    )