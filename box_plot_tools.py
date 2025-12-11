import os
from typing import Optional, Union

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from io import BytesIO

import streamlit as st


# ------------------------------------------------------
# Core plotting logic (no UI)
# ------------------------------------------------------

def generate_box_plot_figure(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    plot_title: str,
    x_label: str,
    y_label: str,
) -> plt.Figure:
    """
    Create a box + strip plot figure (no GUI, no saving, just returns a Matplotlib Figure).
    """
    if not all([x_column, y_column, plot_title, x_label, y_label]):
        raise ValueError("x_column, y_column, plot_title, x_label, and y_label must all be provided")

    sns.set(style="whitegrid")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Order categories by mean of y
    order = data.groupby(x_column)[y_column].mean().sort_values(ascending=False).index

    # Color palette
    box_palette = ["#64D273", "#9B91F9", "#ED695D", "#FFD750"]
    box_colors = {key: box_palette[i % len(box_palette)] for i, key in enumerate(order)}

    # Boxplot
    box_plot = sns.boxplot(
        x=x_column,
        y=y_column,
        data=data,
        order=order,
        showmeans=True,
        meanprops={
            "marker": "*",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 15,
        },
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 2},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        medianprops={"color": "black"},
        ax=ax,
    )

    # Color each box edge
    for i, box in enumerate(box_plot.artists):
        box.set_edgecolor(box_colors[order[i]])
        box.set_linewidth(2)

    # Stripplot
    sns.stripplot(
        x=x_column,
        y=y_column,
        data=data,
        order=order,
        jitter=True,
        hue=x_column,
        palette=box_colors,
        edgecolor="none",
        linewidth=0,
        dodge=False,
        legend=False,
        ax=ax,
    )

    # Overlay mean markers
    means = data.groupby(x_column)[y_column].mean().reindex(order)
    for i, mean in enumerate(means):
        ax.plot(i, mean, marker="*", color="black", markersize=15, zorder=10)

    # Titles and labels
    ax.set_title(plot_title, fontsize=16, fontweight="bold", fontname="Urbane Rounded")
    ax.set_xlabel(x_label, fontsize=14, fontname="Rubik")
    ax.set_ylabel(y_label, fontsize=14, fontname="Rubik")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    # Background and grid
    ax.set_facecolor("#E3E3E2")
    ax.grid(color="white", which="both", linestyle="-", linewidth=1)
    ax.xaxis.grid(True, linestyle="-", linewidth=1, color="white")

    # Optional: adjust y-axis tick spacing (approx half the default step)
    yticks = ax.get_yticks()
    if len(yticks) > 1:
        step = yticks[1] - yticks[0]
        ax.yaxis.set_major_locator(MultipleLocator(step / 2))

    fig.subplots_adjust(bottom=0.2)

    return fig


def generate_box_plot_from_excel(
    file: Union[str, bytes, bytearray],
    x_column: str,
    y_column: str,
    plot_title: str,
    x_label: str,
    y_label: str,
    sheet: Optional[Union[str, int]] = None,
) -> plt.Figure:
    """
    Convenience function:
    - Load Excel (path or bytes)
    - Optionally choose a sheet (by name or index; default = first sheet)
    - Build DataFrame
    - Return a Matplotlib Figure with the box plot
    """
    # Create an ExcelFile object so we can choose sheet
    if isinstance(file, (bytes, bytearray)):
        excel = pd.ExcelFile(BytesIO(file))
    else:
        excel = pd.ExcelFile(file)

    # Default to first sheet if none specified
    if sheet is None:
        sheet = excel.sheet_names[0]

    data = excel.parse(sheet_name=sheet)

    return generate_box_plot_figure(
        data=data,
        x_column=x_column,
        y_column=y_column,
        plot_title=plot_title,
        x_label=x_label,
        y_label=y_label,
    )


def save_box_plot_png(
    fig: plt.Figure,
    output_path: str,
) -> None:
    """
    Save a Matplotlib figure as a PNG file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------
# Streamlit UI wrapper for the Box Plot tool
# ------------------------------------------------------

class BoxPlotUI:
    """Streamlit UI for the Box Plot builder (Data & Graphs section)."""

    @classmethod
    def render(cls):
        st.subheader("Box plot builder")

        st.markdown(
            """
            Upload an Excel file, choose a **sheet**, pick your X and Y columns,
            and Rootweiler will generate a publication-ready box + strip plot.
            """
        )

        uploaded = st.file_uploader(
            "Upload Excel file",
            type=["xlsx", "xls"],
            key="boxplot_excel_uploader",
        )

        if uploaded is None:
            st.info("Upload an Excel file to begin.")
            return

        # Read sheet names
        try:
            excel = pd.ExcelFile(uploaded)
        except Exception as e:
            st.error(f"Could not read Excel file: {e}")
            return

        # Let the user choose the sheet
        sheet_name = st.selectbox(
            "Select worksheet",
            options=excel.sheet_names,
            index=0,
        )

        # Parse selected sheet into DataFrame
        try:
            df = excel.parse(sheet_name=sheet_name)
        except Exception as e:
            st.error(f"Could not read sheet '{sheet_name}': {e}")
            return

        if df.empty:
            st.warning("Selected sheet appears to be empty.")
            return

        st.markdown("#### Data preview")
        st.dataframe(df.head(), use_container_width=True)

        columns = df.columns.tolist()
        if len(columns) < 2:
            st.warning("Need at least two columns in the sheet to build a box plot.")
            return

        st.markdown("#### Plot configuration")

        c1, c2 = st.columns(2)
        with c1:
            x_column = st.selectbox("X-axis (group/category)", options=columns)
            x_label = st.text_input("X-axis label", value=x_column)

        with c2:
            y_column = st.selectbox("Y-axis (numeric)", options=columns)
            y_label = st.text_input("Y-axis label", value=y_column)

        plot_title = st.text_input("Plot title", value=f"Box plot of {y_column} by {x_column}")

        if st.button("Generate box plot", type="primary"):
            try:
                fig = generate_box_plot_figure(
                    data=df,
                    x_column=x_column,
                    y_column=y_column,
                    plot_title=plot_title,
                    x_label=x_label,
                    y_label=y_label,
                )
            except Exception as e:
                st.error(f"Could not generate plot: {e}")
                return

            st.markdown("#### Box plot")
            st.pyplot(fig, use_container_width=True)

            # Prepare PNG for download
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)

            st.download_button(
                "Download plot as PNG",
                data=buf,
                file_name="rootweiler_box_plot.png",
                mime="image/png",
            )

            # Close the figure to free memory on rerun
            plt.close(fig)
