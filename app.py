from __future__ import annotations

from pathlib import Path
import re
from typing import Optional

import pandas as pd

# --------------------------------------------------
# Optional imports with graceful fallback
# --------------------------------------------------
# This dashboard supports both:
# 1. Streamlit interactive dashboard mode
# 2. Console preview fallback mode
#
# It is designed to run safely in:
# - local terminals
# - notebooks
# - sandboxed environments
# - CI pipelines
# - Streamlit Cloud deployments
# --------------------------------------------------

STREAMLIT_AVAILABLE = True
PLOTLY_AVAILABLE = True

try:
    import streamlit as st
except ModuleNotFoundError:
    STREAMLIT_AVAILABLE = False

try:
    import plotly.express as px
except ModuleNotFoundError:
    PLOTLY_AVAILABLE = False


# --------------------------------------------------
# Safe path handling
# --------------------------------------------------
# Some environments (notebooks/sandboxes) do not
# define __file__. We safely fall back to cwd().
# --------------------------------------------------

try:
    APP_DIR = Path(__file__).resolve().parent
except NameError:
    APP_DIR = Path.cwd()

DATA_DIR = APP_DIR / "data"


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def clean_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name)).strip().lower()



def find_column(df: pd.DataFrame, keywords: list[str]) -> Optional[str]:
    """Return the first column whose name contains a keyword."""

    cols = list(df.columns)
    normalized = {c: clean_name(c) for c in cols}

    for kw in keywords:
        kw = kw.lower()

        for col, norm in normalized.items():
            if kw in norm:
                return col

    return None



def to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("-", "", regex=False)
        .str.strip(),
        errors="coerce",
    )



def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)



def get_csv_files() -> list[Path]:
    if not DATA_DIR.exists():
        return []

    return sorted(DATA_DIR.glob("*.csv"))



def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {}

    year_col = find_column(df, ["year", "acad year", "academic year"])
    uni_col = find_column(df, ["university", "institution"])
    degree_col = find_column(df, ["degree", "course", "field of study", "programme", "program"])
    salary_col = find_column(df, ["median gross monthly salary", "gross monthly salary", "salary", "median salary"])
    employment_col = find_column(df, ["full time employment rate", "employment rate", "employment"])
    cohort_col = find_column(df, ["cohort", "graduate cohort"])
    sector_col = find_column(df, ["industry", "sector"])

    mapping_pairs = {
        year_col: "year",
        uni_col: "university",
        degree_col: "degree",
        salary_col: "salary",
        employment_col: "employment_rate",
        cohort_col: "cohort",
        sector_col: "sector",
    }

    for source, target in mapping_pairs.items():
        if source and source not in rename_map:
            rename_map[source] = target

    df = df.rename(columns=rename_map)

    for col in ["year", "salary", "employment_rate"]:
        if col in df.columns:
            df[col] = to_numeric_series(df[col])

    return df


# --------------------------------------------------
# Console fallback mode
# --------------------------------------------------

def run_console_preview(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("Singapore Graduate Employment Dashboard")
    print("Console Preview Mode")
    print("=" * 60)

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    if "salary" in df.columns:
        print(f"Average Salary: {df['salary'].mean(skipna=True):,.0f}")

    if "employment_rate" in df.columns:
        print(
            f"Average Employment Rate: "
            f"{df['employment_rate'].mean(skipna=True):.1f}"
        )

    print("\nColumns:")
    for col in df.columns:
        print(f"- {col}")

    print("\nSample Data:")
    print(df.head(10).to_string())


# --------------------------------------------------
# Streamlit dashboard mode
# --------------------------------------------------

def run_streamlit_dashboard() -> None:
    st.set_page_config(
        page_title="Singapore Graduate Employment Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.title("Singapore Graduate Employment Dashboard")

    st.caption(
        "A simple, interactive dashboard for graduate employment insights."
    )

    csv_files = get_csv_files()

    if not csv_files:
        st.error(
            "No CSV files found in the `data/` folder. "
            "Place CSV files inside a data directory."
        )
        st.stop()

    selected_file = st.sidebar.selectbox(
        "Choose dataset",
        csv_files,
        format_func=lambda p: p.name,
    )

    df = prepare_frame(load_csv(selected_file))

    st.sidebar.header("Filters")

    filtered = df.copy()

    for col in ["year", "university", "degree", "cohort", "sector"]:
        if col in filtered.columns:
            values = sorted(
                [v for v in filtered[col].dropna().unique().tolist()]
            )

            if not values:
                continue

            chosen = st.sidebar.multiselect(
                col.replace("_", " ").title(),
                values,
                default=values,
            )

            filtered = filtered[filtered[col].isin(chosen)]

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    metric_cols = st.columns(4)

    metric_cols[0].metric("Records", f"{len(filtered):,}")

    if "salary" in filtered.columns:
        metric_cols[1].metric(
            "Avg Salary",
            f"{filtered['salary'].mean(skipna=True):,.0f}",
        )
    else:
        metric_cols[1].metric("Avg Salary", "N/A")

    if "employment_rate" in filtered.columns:
        metric_cols[2].metric(
            "Avg Employment",
            f"{filtered['employment_rate'].mean(skipna=True):.1f}",
        )
    else:
        metric_cols[2].metric("Avg Employment", "N/A")

    if "degree" in filtered.columns and not filtered.empty:
        top_degree = (
            filtered.groupby("degree")
            .size()
            .sort_values(ascending=False)
            .index[0]
        )

        metric_cols[3].metric("Top Degree", str(top_degree))
    else:
        metric_cols[3].metric("Top Degree", "N/A")

    st.divider()

    # --------------------------------------------------
    # Charts
    # --------------------------------------------------

    if not PLOTLY_AVAILABLE:
        st.warning(
            "Plotly is not installed. Install plotly to enable charts."
        )
    else:
        left, right = st.columns(2)

        with left:
            st.subheader("Salary by Degree")

            if {"degree", "salary"}.issubset(filtered.columns):
                degree_salary = (
                    filtered.dropna(subset=["degree", "salary"])
                    .groupby("degree", as_index=False)["salary"]
                    .mean()
                    .sort_values("salary", ascending=False)
                    .head(10)
                )

                fig = px.bar(
                    degree_salary,
                    x="salary",
                    y="degree",
                    orientation="h",
                    text_auto=".0f",
                )

                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Missing degree or salary columns.")

        with right:
            st.subheader("Employment by Degree")

            if {"degree", "employment_rate"}.issubset(filtered.columns):
                degree_emp = (
                    filtered.dropna(subset=["degree", "employment_rate"])
                    .groupby("degree", as_index=False)["employment_rate"]
                    .mean()
                    .sort_values("employment_rate", ascending=False)
                    .head(10)
                )

                fig = px.bar(
                    degree_emp,
                    x="employment_rate",
                    y="degree",
                    orientation="h",
                    text_auto=".1f",
                )

                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Missing degree or employment rate columns.")

    st.divider()

    st.subheader("Filtered Dataset")
    st.dataframe(filtered, use_container_width=True, height=400)

    st.caption(
        "Deploy with Streamlit Community Cloud using app.py as the entrypoint."
    )


# --------------------------------------------------
# Main entrypoint
# --------------------------------------------------

def main() -> None:
    csv_files = get_csv_files()

    if not csv_files:
        print(
            "No CSV files found inside the data folder. "
            f"Expected path: {DATA_DIR}"
        )
        return

    df = prepare_frame(load_csv(csv_files[0]))

    if STREAMLIT_AVAILABLE:
        run_streamlit_dashboard()
    else:
        print(
            "\n[INFO] Streamlit is not installed. "
            "Running console preview instead.\n"
        )
        run_console_preview(df)


# --------------------------------------------------
# Tests
# --------------------------------------------------

def _test_clean_name() -> None:
    assert clean_name("  Hello   World  ") == "hello world"
    assert clean_name("TEST") == "test"



def _test_find_column() -> None:
    sample = pd.DataFrame(columns=["Year", "Median Salary"])

    assert find_column(sample, ["year"]) == "Year"
    assert find_column(sample, ["salary"]) == "Median Salary"
    assert find_column(sample, ["employment"]) is None



def _test_to_numeric_series() -> None:
    series = pd.Series(["4,500", "95.2%", "-"])

    result = to_numeric_series(series)

    assert result.iloc[0] == 4500
    assert result.iloc[1] == 95.2
    assert pd.isna(result.iloc[2])



def _test_prepare_frame() -> None:
    sample = pd.DataFrame(
        {
            "Year": [2024],
            "Median Gross Monthly Salary": ["4,500"],
            "Full Time Employment Rate": ["95.2%"],
        }
    )

    result = prepare_frame(sample)

    assert "salary" in result.columns
    assert "employment_rate" in result.columns
    assert result["salary"].iloc[0] == 4500
    assert result["employment_rate"].iloc[0] == 95.2



def _test_app_dir_exists() -> None:
    assert isinstance(APP_DIR, Path)


if __name__ == "__main__":
    _test_clean_name()
    _test_find_column()
    _test_to_numeric_series()
    _test_prepare_frame()
    _test_app_dir_exists()

    main()
