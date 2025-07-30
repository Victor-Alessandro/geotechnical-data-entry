import os
import numpy as np
import streamlit as st
import pandas as pd
import json
import re

from typing import List, Dict, Any, Tuple
from collections import namedtuple
from rapidfuzz import fuzz
from math import log
from option import Some

# ==============================================================================
# 1. INITIALIZATION & DATA STRUCTURES
# ==============================================================================

# Encapsulate keyword data into a namedtuple for clarity and consistency.
Keyword = namedtuple('Keyword', ['word', 'weight', 'similarity_requirement'])

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    session_defaults = {
        'etude_sol_uploaded': False,
        'document_tables': [],
        'table_ratings': {}
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ==============================================================================
# 2. DATA PARSING & PREPARATION FUNCTIONS
# ==============================================================================

@st.cache_data
def parse_tables_from_json(json_data: Dict[str, Any]) -> List[pd.DataFrame]:
    """Parse table elements from the JSON structure into pandas DataFrames."""
    if 'elements' not in json_data:
        return []
        
    dataframes = []
    for element in json_data['elements']:
        if element.get("type") == "table" and element.get("table"):
            df = parse_single_table_to_dataframe(element)
            if not df.empty:
                dataframes.append(df)
    return dataframes


def parse_single_table_to_dataframe(table_element: Dict[str, Any]) -> pd.DataFrame:
    """Parse a single table element into a pandas DataFrame."""
    cells = table_element.get("table", {}).get("cells", [])
    if not cells:
        return pd.DataFrame()
    
    valid_cells = [cell for cell in cells if cell.get("rows") and cell.get("cols")]
    if not valid_cells:
        return pd.DataFrame()
    
    max_row = max(max(cell.get("rows", [0])) for cell in valid_cells)
    max_col = max(max(cell.get("cols", [0])) for cell in valid_cells)
    
    matrix = np.full((max_row + 1, max_col + 1), None, dtype=object)
    
    for cell in cells:
        content = cell.get("content", "")
        for row in cell.get("rows", []):
            for col in cell.get("cols", []):
                if 0 <= row <= max_row and 0 <= col <= max_col:
                    matrix[row, col] = content
    
    header_rows_indices = sorted(list(set(
        row for cell in cells if cell.get("is_header") for row in cell.get("rows", [])
    )))
    
    if not header_rows_indices and matrix.shape[0] > 0:
        header_rows_indices = [0]
    
    headers = [f"Colonne {i+1}" for i in range(matrix.shape[1])]
    
    if header_rows_indices:
        header_matrix = matrix[header_rows_indices, :]
        if len(header_rows_indices) > 1:
            headers = [
                " | ".join(filter(None, [str(h) for h in header_matrix[:, col_idx] if h is not None]))
                or f"Colonne {col_idx + 1}"
                for col_idx in range(header_matrix.shape[1])
            ]
        else:
            headers = [str(h) if h is not None else f"Colonne {i+1}" for i, h in enumerate(header_matrix[0])]
    
    data_rows = [i for i in range(matrix.shape[0]) if i not in header_rows_indices]
    
    if not data_rows:
        data = np.full((0, len(headers)), None, dtype=object)
    else:
        data = matrix[data_rows, :]
    
    if data.shape[1] != len(headers):
        if data.shape[1] < len(headers):
            padding = np.full((data.shape[0], len(headers) - data.shape[1]), None, dtype=object)
            data = np.hstack([data, padding])
        else:
            data = data[:, :len(headers)]
    
    df = pd.DataFrame(data, columns=headers)
    
    df = df.fillna("")
    
    return fix_duplicate_columns(df)

def fix_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fix duplicate column names by adding suffixes."""
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [f"{dup}_{i}" if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df

# ==============================================================================
# 3. CORE FILTERING & SCORING LOGIC
# ==============================================================================

def is_convertible_to_number(x: Any) -> bool:
    """Check if a value can be converted to a number, handling comma decimals."""
    if isinstance(x, (int, float)):
        return True
    try:
        float(str(x).replace(',', '.'))
        return True
    except (ValueError, TypeError):
        return False

def dataframe_has_numeric(df: pd.DataFrame) -> bool:
    """Check if any cell in the DataFrame contains a numeric value."""
    return df.apply(lambda s: s.apply(is_convertible_to_number)).any().any()

# maybe use WRatio instead
def score_dataframes_by_headers(
    dataframes: List[pd.DataFrame], 
    header_keywords: List[Keyword]
) -> List[Dict[str, Any]]:
    """Score DataFrames based on fuzzy matching of headers and return sorted results."""
    def process_header(header):
        return next(
            (
                Some((keyword, token, similarity))
                for keyword in header_keywords
                for token in set(header.split())
                if (similarity := fuzz.WRatio(keyword.word.lower(), token)) >= keyword.similarity_requirement
            ),
            None,
        )

    results = []
    for i, df in enumerate(dataframes):
        headers = [re.sub(r'[\(\)\[\]\{\}*_]', ' ', h).lower() for h in df.columns if h]

        total_score = 0
        number_of_words = 0
        matched_words_details = set()

        for header in headers:
            result = process_header(header)
            number_of_words += len(header.split())
            if result:
                keyword, token, similarity = result.unwrap()
                total_score += keyword.weight
                matched_words_details.add(keyword.word)

        if total_score > 0:
            multiplier = 3
            total_score = multiplier * total_score / log(1 + number_of_words**2)
            results.append({
                'df_index': i,
                'df': df,
                'header_score': total_score,
                'matched_headers': list(matched_words_details)
            })

    results.sort(key=lambda x: x['header_score'], reverse=True)
    return results

def score_dataframe_by_bag_of_words(
    df: pd.DataFrame, 
    soil_keywords: List[Keyword]
) -> Tuple[float, List[str], Dict[Tuple[int, int], List[str]]]:
    """
    Calculate a normalized, weighted score based on matching words in the DataFrame's body.
    The score is normalized by the number of non-numeric cells to reward textually-dense matches.
    Returns the score, a list of matched words, and a dictionary of match locations.
    """
    total_weighted_score = 0.0
    matched_target_words = set()
    number_of_options_tested = 0
    match_locations = {}  # Key: (row_idx, col_idx), Value: list of matched keyword.word

    for r_idx in range(df.shape[0]):
        for c_idx in range(df.shape[1]):
            cell_value = df.iat[r_idx, c_idx]
            if not is_convertible_to_number(cell_value) and str(cell_value).strip():
                cell_content = str(cell_value).lower()
                number_of_options_tested += len(cell_content.split())
                
                found_in_cell = set()
                for keyword in soil_keywords:
                    if keyword.word not in found_in_cell and fuzz.partial_ratio(keyword.word.lower(), cell_content) >= keyword.similarity_requirement:
                        total_weighted_score += keyword.weight
                        matched_target_words.add(keyword.word)
                        found_in_cell.add(keyword.word)
                        
                        # Record the match location
                        if (r_idx, c_idx) not in match_locations:
                            match_locations[(r_idx, c_idx)] = []
                        match_locations[(r_idx, c_idx)].append(keyword.word)

    if number_of_options_tested == 0:
        return 0.0, [], {}

    multiplier = 3
    normalized_score = multiplier * total_weighted_score / log(1 + number_of_options_tested**2)
    
    return normalized_score, list(matched_target_words), match_locations

# ==============================================================================
# 4. STREAMLIT UI & MAIN APPLICATION FLOW
# ==============================================================================

def display_main_interface():
    """Main UI for displaying results and interactions after file upload."""
    
    st.header("⚙️ Configuration des Filtres")
   
    positive_keywords = [
        Keyword('formation', 1, 75), Keyword('nature', 1, 75), Keyword('sol', 1, 75), 
        Keyword('horizon', 1, 75), Keyword('couche', 1, 70), Keyword('faciès', 1, 70), 
        Keyword('profondeur', 1, 75), Keyword('prof', 1, 75), Keyword('épaisseur', 1, 70), 
        Keyword('mpa', 1, 85), Keyword('pl', 2.8, 85), Keyword('pi', 2.8, 85), 
        Keyword('pression', 1, 75), Keyword('kp', 2.5, 85), Keyword('courbe', 1.0, 75),
        Keyword('alfa', 2.0, 90), Keyword('α', 2.0, 90),
        Keyword('em', 0.5, 80), Keyword( 'terrain', 1, 75)]

    negative_keywords = [
        Keyword('avancement', -15, 80), Keyword('vitesse', -15, 80), Keyword('injection', -15, 80),
        Keyword('rotation', -15, 80), Keyword('fluage', -15, 80), Keyword('sondage', -10, 80)
    ]

    soil_type_keywords = [
        Keyword('sable', 1, 85), Keyword('alluvions', 1, 85), Keyword('remblais', 1, 85), 
        Keyword('tuffeau', 1, 85), Keyword('craie', 1, 85), Keyword('argile', 1, 85), 
        Keyword('limons', 1, 85), Keyword('marne', 1, 85), Keyword('graves', 1, 85), 
        Keyword('calcaire', 1, 85), Keyword('roche', 1, 85),
        Keyword('retenue', 3.0, 80), Keyword('max', 1, 90), Keyword('min', 1, 90)
    ]
    
    with st.spinner("Analyse des tableaux en cours..."):
        header_scored_tables = score_dataframes_by_headers(st.session_state.document_tables, positive_keywords + negative_keywords)
        
        final_results = []
        for table_data in header_scored_tables:
            if dataframe_has_numeric(table_data['df']):
                bow_score, matched_soil_types, match_details = score_dataframe_by_bag_of_words(table_data['df'], soil_type_keywords)
                
                final_score = table_data['header_score'] * log(1 + bow_score)
                
                table_data['bow_score'] = bow_score
                table_data['matched_soil_types'] = matched_soil_types
                table_data['final_score'] = final_score
                table_data['match_details'] = match_details # Store match locations
                final_results.append(table_data)

        final_results.sort(key=lambda x: x['final_score'], reverse=True)

    st.success(f"Analyse terminée. {len(final_results)} tableaux pertinents trouvés sur {len(st.session_state.document_tables)} au total.")
    
    st.header("🏆 Meilleurs Résultats")
    
    num_to_show = st.slider(
        "Nombre de tableaux à afficher :", 
        min_value=1, 
        max_value=max(1, len(final_results)), 
        value=min(5, len(final_results) or 1), 
        step=1
    )

    top_results = final_results[:num_to_show]

    if not top_results:
        st.warning("Aucun tableau ne correspond aux critères de recherche après filtrage.")
    else:
        for result in top_results:
            df_idx = result['df_index']
            df = result['df']
            match_details = result['match_details']
            
            with st.container(border=True):
                st.subheader(f"Tableau Pertinent (Index: {df_idx})")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Score Final", f"{result['final_score']:.2f}")
                    st.write("**Paramètres d'en-tête trouvés :**")
                    st.write(", ".join(result['matched_headers']) or "Aucun")
                with col2:
                    st.metric("Score Contenu (Pondéré & Normalisé)", f"{result['bow_score']:.2f}")
                    st.write("**Types de sol trouvés :**")
                    st.write(", ".join(result['matched_soil_types']) or "Aucun")
                
                st.write("---")
                
                # --- Display Original and Highlighted DataFrames ---
                st.markdown("**Tableau Original**")
                st.dataframe(df, use_container_width=True)

                if match_details:
                    st.markdown("**Tableau avec Correspondances de Mots Clés**")
                    
                    # Create a display-only copy to modify with match info
                    display_df = df.copy().astype(str)
                    
                    # Add match information to the cells
                    for (r, c), words in match_details.items():
                        original_value = display_df.iat[r, c]
                        matched_str = ", ".join(words)
                        display_df.iat[r, c] = f"{original_value} [MATCH: {matched_str}]"
                    
                    # Styler function to highlight cells with matches
                    def highlight_matches(val):
                        if '[MATCH:' in val:
                            return 'background-color: yellow; color: black;'
                        return ''
                    
                    st.dataframe(display_df.style.applymap(highlight_matches), use_container_width=True)

                # --- Rating Slider ---
                rating_key = f"rating_{df_idx}"
                st.slider(
                    label=f"Attribuer une note de pertinence au Tableau {df_idx}",
                    min_value=0,
                    max_value=10,
                    key=rating_key,
                    value=st.session_state.table_ratings.get(df_idx, 0)
                )
                st.session_state.table_ratings[df_idx] = st.session_state[rating_key]

    with st.expander("📂 Afficher tous les tableaux extraits"):
        display_all_tables_explorer()

def display_all_tables_explorer():
    """Fragment for showing all tables and allowing selection."""
    
    @st.fragment
    def show_all_tables_fragment():
        st.info(f"Voici les en-têtes des {len(st.session_state.document_tables)} tableaux extraits par l'outil de reconnaissance optique.")
        
        column_indices_map = {}
        for idx, df in enumerate(st.session_state.document_tables):
            cols_tuple = tuple(df.columns)
            if cols_tuple not in column_indices_map:
                column_indices_map[cols_tuple] = []
            column_indices_map[cols_tuple].append(idx)
        
        keys_to_dfs = list(column_indices_map.keys())
        if not keys_to_dfs:
            st.warning("Aucune structure de tableau à afficher.")
            return

        max_len = max(len(key) for key in keys_to_dfs) if keys_to_dfs else 0
        normalized_list = [list(inner_list) + [''] * (max_len - len(inner_list)) for inner_list in keys_to_dfs]
        structures_df = pd.DataFrame(normalized_list)

        selection = st.dataframe(
            structures_df, use_container_width=True, hide_index=True,
            on_select='rerun', selection_mode='multi-row'
        )

        if selection and 'rows' in selection.get('selection', {}):
            selected_rows = selection['selection']['rows']
            if st.button("🔍 Afficher les tableaux sélectionnés"):
                if not selected_rows:
                    st.warning("Veuillez sélectionner des structures de tableau dans la liste ci-dessus.")
                else:
                    selected_indices = []
                    for row_idx in selected_rows:
                        clean_key = tuple(filter(lambda a: a != "", normalized_list[row_idx]))
                        if clean_key in column_indices_map:
                            selected_indices.extend(column_indices_map[clean_key])

                    for i, df_idx in enumerate(sorted(list(set(selected_indices)))):
                        st.write(f"**Tableau (Index d'origine {df_idx}):**")
                        st.dataframe(st.session_state.document_tables[df_idx], use_container_width=True)
    
    show_all_tables_fragment()

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Analyseur de Tableaux de Documents", page_icon="📊", layout="wide")
    st.title("📊 Analyseur de Tableaux de Documents")
    
    initialize_session_state()
    
    if not st.session_state.etude_sol_uploaded:
        st.header("📁 Étape 1 : Charger le document JSON")
        uploaded_json = st.file_uploader(
            "Choisir le fichier d'étude de sol (format JSON)",
            type=['json'],
            key="etude_sol"
        )
        if uploaded_json:
            try:
                json_data = json.load(uploaded_json)
                tables = parse_tables_from_json(json_data)
                if tables:
                    st.session_state.document_tables = tables
                    st.session_state.etude_sol_uploaded = True
                    st.success(f"{len(tables)} tableaux ont été chargés avec succès !")
                    st.rerun()
                else:
                    st.warning("Aucun tableau n'a été trouvé dans le fichier JSON fourni.")
            except Exception as e:
                st.error(f"Erreur lors du traitement du fichier : {e}")
    else:

        tab1, tab2, tab3, tab4 = st.tabs(["visible score tableaux", "empty", "empty", "empty"])

        with tab1:
            st.info("Un fichier est déjà chargé. Pour en analyser un nouveau, veuillez réinitialiser.")
            if st.button("🔄 Réinitialiser et charger un nouveau fichier"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
            
            st.markdown("---")
            display_main_interface()

if __name__ == "__main__":
    main()
