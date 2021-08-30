import shelve
from pathlib import Path

import streamlit as st
from PIL import Image

from mavis.pdutils import image_columns, update
from mavis.shelveutils import current_project


def gallery():
    st.markdown("")
    col_selection, col_options = st.beta_columns(2)
    project = current_project()
    with shelve.open(project) as d:
        df = d["df"]
    with col_selection:
        columns = st.multiselect("⋮⋮⋮", image_columns(df))
    if columns:
        with col_options:
            st.markdown("   ")
            display_options = st.beta_expander("Gallery Options")
            with display_options:
                n_columns = len(columns)
                columns_per_column = st.slider("Gallery Width", 1, 10, max(4 - len(columns), 1))
                max_items_per_page = st.slider("Images per Page", 1, 200, 20)
                show_caption = st.checkbox("Show Caption")
                is_selectable = st.checkbox("Flag Images")
                df_filtered = df

                if st.checkbox("Filter"):
                    filter_column = st.selectbox("Select Filter Column", df.columns)
                    filter_values = st.multiselect("Filter By", list(df[filter_column].unique()))
                    if filter_values:
                        df_filtered = df[df[filter_column].isin(filter_values)]
                    df = df_filtered

                order_column = st.selectbox("Order display by column", df.columns)
                if order_column:
                    desc = st.checkbox("Descending", False, key="Desc.Img.Gallery")
                    df = df.sort_values(order_column, ascending=not desc).reset_index(drop=True)
                    st.info(f"Ordered by {order_column}")
                if is_selectable:
                    flag_column = st.text_input("Flag Column", "Flag")
                    flag_value = st.text_input("Flag Value")

        with st.beta_expander("Gallery"):
            paths = df[columns].dropna()
            items_per_page = (min(len(paths), max_items_per_page))
            n_pages = (len(paths) - 1) // items_per_page + 1
            page_number = (st.slider("Page", 1, n_pages) - 1) if n_pages > 1 else 0

            min_index = page_number * items_per_page
            max_index = min(min_index + items_per_page, len(paths)-1)

            selections = {}
            current_ind = min_index
            while current_ind <= max_index:
                current_column = 0
                ind_layout = st.beta_columns(columns_per_column)
                col_layout = st.beta_columns(n_columns * columns_per_column)
                for i in range(columns_per_column):
                    df_index = paths.index[current_ind]

                    if is_selectable:
                        with ind_layout[i]:
                            selections[df_index] = st.checkbox(f"{df_index}", flag_column in df and df[flag_column][df_index] == flag_value)

                    for j in range(n_columns):
                        with col_layout[current_column]:
                            col_name = columns[j]
                            path = paths[col_name][df_index]
                            caption = f'{df_index}: {Path(path).name}' if show_caption else ""
                            st.image(Image.open(path), use_column_width=True, caption=caption)
                            current_column += 1
                    current_ind += 1
                    if current_ind > max_index:
                        break

        if is_selectable:
            with display_options:
                if flag_column not in df:
                    df[flag_column] = None
                if st.button("Apply"):
                    for k, v in selections.items():
                        df.loc[k, flag_column] = flag_value if v else df.loc[k, flag_column]
                    update(df, project)
                    st.info("Updated Flags.")


def paginator(label, items, items_per_page=10, on_sidebar=True):
    # Figure out where to display the paginator
    if on_sidebar:
        location = st.sidebar.empty()
    else:
        location = st.empty()

    # Display a pagination selectbox in the specified location.
    items = list(items)
    n_pages = (len(items) - 1) // items_per_page + 1
    page_number = (location.slider("Page", 1, n_pages, key=label) - 1) if n_pages > 1 else 0

    # Iterate over the items in the page to let the user display them.
    min_index = page_number * items_per_page
    max_index = min_index + items_per_page
    import itertools
    return itertools.islice(enumerate(items), min_index, max_index)