import os
import numpy as np
import streamlit as st
import pandas as pd
import re
import fitz
import tempfile

from aryn_sdk.partition import partition_file
from typing import List, Dict, Any, Tuple, Optional, Literal, ClassVar
from pydantic import BaseModel, model_validator, conlist
from pydantic_ai import Agent

# abstract nonsense
from abc import ABC

from collections import namedtuple
from rapidfuzz import fuzz
from math import log

# ==============================================================================
# 1. DATA STRUCTURES & CONFIGURATION
# ==============================================================================

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    session_defaults = {
        'etude_sol_uploaded': False,
        'document_tables': [],
        'table_ratings': {},
        'selected_tables': pd.DataFrame,
        'pertinent_table': pd.DataFrame,
        'raccourci_loaded': False
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

Keyword = namedtuple('Keyword', ['word', 'weight', 'similarity_requirement'])

# I might abstract this to work on any namedtuple and then curry to define create_keywords
@st.cache_data
def create_keywords(keyword_tuples: List[Tuple[str, float, int]]) -> List[Keyword]:
    """Convert list of tuples to list of Keyword namedtuples."""
    return [Keyword(word, weight, similarity) for word, weight, similarity in keyword_tuples]

@st.cache_data
def get_keyword_sets() -> Dict[str, List[Keyword]]:
    """Define all keyword sets used for analysis."""
    positive_tuples = [
        ('formation', 1, 75), ('nature', 1, 75), ('sol', 1, 75), 
        ('horizon', 1, 75), ('couche', 1, 70), ('faciès', 1, 70), 
        ('profondeur', 1, 75), ('prof', 1, 75), ('épaisseur', 1, 70), 
        ('mpa', 1, 85), ('pl', 2.8, 85), ('pi', 2.8, 85), 
        ('pression', 1, 75), ('kp', 2.5, 85), ('courbe', 1.0, 75),
        ('alfa', 2.0, 90), ('α', 2.0, 90),
        ('em', 0.5, 80), ('terrain', 1, 75)
    ]
    
    negative_tuples = [
        ('avancement', -15, 80), ('vitesse', -15, 80), ('injection', -15, 80),
        ('rotation', -15, 80), ('fluage', -15, 80), ('sondage', -10, 80)
    ]
    
    soil_type_tuples = [
        ('sable', 1, 85), ('alluvions', 1, 85), ('remblais', 1, 85), 
        ('tuffeau', 1, 85), ('craie', 1, 85), ('argile', 1, 85), 
        ('limons', 1, 85), ('marne', 1, 85), ('graves', 1, 85), 
        ('calcaire', 1, 85), ('roche', 1, 85),
        ('retenue', 3.0, 80), ('max', 1, 90), ('min', 1, 90)
    ]
    
    return {
        'positive': create_keywords(positive_tuples),
        'negative': create_keywords(negative_tuples),
        'soil_types': create_keywords(soil_type_tuples)
    }

# ==============================================================================
# 2. DATA PARSING & PREPARATION
# ==============================================================================


@st.cache_data
def parse_tables_from_json(json_data: Dict[str, Any]) -> List[Tuple[pd.DataFrame, int]]:
    """Parse table elements from the JSON structure into pandas DataFrames."""
    if 'elements' not in json_data:
        return []
        
    dataframes = []
    for element in json_data['elements']:
        if element.get("type") == "table" and element.get("table"):
            df = parse_single_table_to_dataframe(element)
            if not df.empty:
                page_number = element.get('properties', {}).get('page_number', 0)
                dataframes.append((df, page_number))
    
    return dataframes


def parse_single_table_to_dataframe(table_element: Dict[str, Any]) -> pd.DataFrame:
    """Parse a single table element into a pandas DataFrame."""
    cells = table_element.get("table", {}).get("cells", [])
    if not cells:
        return pd.DataFrame()
    
    matrix, header_rows_indices = process_table_cells(cells)
    if matrix.size == 0:
        return pd.DataFrame()
    
    return create_dataframe_from_matrix(matrix, header_rows_indices)


def process_table_cells(cells: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[int]]:
    """Process table cells and extract matrix and header information."""
    valid_cells = [cell for cell in cells if cell.get("rows") and cell.get("cols")]
    if not valid_cells:
        return np.array([]), []
    
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
    
    return matrix, header_rows_indices



def create_dataframe_from_matrix(matrix: np.ndarray, header_rows_indices: List[int]) -> pd.DataFrame:
    """Create a pandas DataFrame from matrix with proper headers."""
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



def truncate_colnames(cols, max_len=25):
    new_names, counts = [], {}
    for col in cols:
        # extract unit in parentheses, if any
        m    = re.search(r'\s*(\([^)]*\))\s*$', col)
        unit = m.group(1) if m else ''
        base = re.sub(r'\s*\([^)]*\)\s*$', '', col)
        # truncate base so total ≤ max_len
        avail     = max_len - len(unit)
        base_trunc = base[:avail].rstrip() if avail>0 else base[:max_len]
        new       = f"{base_trunc}{unit}"
        # disambiguate duplicates
        counts[new] = counts.get(new, 0) + 1
        if counts[new] > 1:
            new = f"{new}_{counts[new]}"
        new_names.append(new)
    return new_names



class GeotechnicalBaseModel(BaseModel, ABC):
    """Base class for geotechnical data models with common validation"""
    
    @model_validator(mode='after')
    def validate_list_lengths(self):
        """Ensure all non-None lists have the same length by padding shorter ones with filler values."""

        # Attention.
        filler_values = [float('nan'), 'Neutralisé']
        
        list_fields = self._get_list_fields()
        non_none_lists = [(name, getattr(self, name)) for name in list_fields if getattr(self, name) is not None]

        if not non_none_lists:
            return self

        max_len = max(len(lst) for _, lst in non_none_lists)

        for name, lst in non_none_lists:
            current_len = len(lst)
            if current_len < max_len:
                # Try to infer the type of the list contents
                filler = self._infer_filler(lst, filler_values)
                padding = [filler] * (max_len - current_len)
                setattr(self, name, lst + padding)

        return self
    
    def _get_list_fields(self) -> List[str]:
        """Get all fields that are lists"""
        return [field_name for field_name, field_info in self.model_fields.items() 
                if self._is_list_field(field_info.annotation)]

    def _infer_filler(self, lst: list[Any], filler_values: list[Any]) -> Any:
        """Infer appropriate filler from known options based on list content."""
        for item in lst:
            if isinstance(item, float):
                return float('nan')
            if isinstance(item, str):
                #I'm too lazy to treat the case with courbe
                return 'Neutralisé'

        return filler_values[0]

    def _is_list_field(self, annotation) -> bool:
        """Check if annotation is a List type"""
        if hasattr(annotation, '__origin__'):
            if annotation.__origin__ is list:
                return True

            if (hasattr(annotation, '__args__') and 
                len(annotation.__args__) > 0 and
                hasattr(annotation.__args__[0], '__origin__') and
                annotation.__args__[0].__origin__ is list):
                return True
        return False # Added return False for cases that don't match any condition
    
    def to_dataframe(self) -> pd.DataFrame: 
        """Convert model to DataFrame using model_fields as columns"""
        data = {}
        
        # Add structured fields
        for field_name in self.model_fields.keys():
            field_value = getattr(self, field_name)
            if isinstance(field_value, list):
                data[field_name] = field_value
        
        return pd.DataFrame(data) if data else pd.DataFrame()



SoilTypes = Literal["Neutralisé", "Sol intermédiaire", "Argile / Limons", "Sables / Graves", "Craie", "Marne / Calcaire", "Roche"]
class FormationGeotechnique(GeotechnicalBaseModel):
    names_of_columns_used_for_extraction: conlist( str, min_length=3, max_length=3 )

    profondeur_de_base: List[float]
    classes_de_sol: List[SoilTypes]
    pression_limite_retenu: List[float]

    sys_prompt: ClassVar[str] = """
Sachant que chaque colonne contiendra le même nombre d'éléments (et valeurs manquants seront NaN). À partir du tableau, extrayez les noms des colonnes qui contient les listes des valeurs correspondant le mieux à:

profondeur_de_base: La colonne contient principalement des valeurs numériques, qu'indiquent les différentes profondeurs
classes_de_sol: La colonne qu'indique la nature des couches de sol, les faciès de la formation géotechnique. Attendez que la colonne contienne des mots similaires à 'sable', 'alluvions', 'remblais', 'craie', 'argile', 'limons', 'marne', 'graves', 'calcaire'. Le 'remblais' est neutralisé. En se basant sur les types de sol qui la forment, veuillez leur associer une des classifications suivantes: "Neutralisé", "Sol intermédiaire", "Argile / Limons", "Sables / Graves", "Craie", "Marne / Calcaire", "Roche"
pression_limite_retenu: La colonne contient principalement des valeurs numériques, qu'indiquent typiquement par pl ou pi la pression limite.

IMPORTANT: 
- Tous les champs sont obligatoires et doivent avoir la même longueur.
- Extraire les données ligne par ligne dans l'ordre exact du tableau original.
- Les 3 colonnes originales seront préservées dans le résultat final pour référence dans la liste names_of_columns_used_for_extraction.

Notez dans l'exemple suivant l'importance d'interpréter le contenu des cellules:

|   |Nature des sols|Epaisseur de la formation|Classification sol EC7|Courbe              |Pl* (kPa)           |qs retenu (kPa)     |Kp max              |
|---|---------------|-------------------------|----------------------|--------------------|--------------------|--------------------|--------------------|
|0  |Remblais en substitution de la depollution|5.5 m                    |                      |                    |                    |Neglige             |                    |
|1  |Remblais       |5.5 m sous les zones purgees, 11 m hors zone purgee|Argiles et limons     |Q1                  |500                 |51                  |                    |
|2  |Sable plus ou moins graveleux, en place ou en remblai|1.5                      |Sols intermediaires. sableux|                    |780                 |                    |1.65                |
|3  |Marno calcaire |                         |Marne et calcaire     |Q4                  |1300                |142                 |1.6                 |

Un résultat acceptable serait:
``` json
{
  "profondeur_de_base": [5.5, 11, 12.5, NaN],
  "classes_de_sol": [
    "Neutralisé",
    "Argile / Limons",
    "Sol intermédiaire",
    "Marne / Calcaire",
  ],
  "pression_limite_retenu": [NaN, 0.5, 0.78, 1.3]
}
"""

class ValeursSupplementaires(GeotechnicalBaseModel):
    names_of_columns_used_for_extraction: conlist( str, min_length=0, max_length=4)

    frottement_limite_retenu: Optional[List[float]] = None
    capacite_de_portance_retenu: Optional[List[float]] = None
    modules_pressiometriques: Optional[List[float]] = None
    coefficients_rheologiques: Optional[List[float]] = None

    sys_prompt: ClassVar[str] = """
Si présentes, trouvez les colonnes qui mieux correspondent à :

frottement_limite_retenu: typiquement indiqués par qs
modules_pressiometriques: typiquement indiqués par Em
coefficients_rheologiques: typiquement indiqués par α ou alpha
capacite_de_portance_retenu: qu'indiquent la pression limite nette équivalente indiqué par kp

IMPORTANT: 
- Ces champs sont optionnels. Mais ceux qui sont présents doivent tous avoir la même longueur.
- Extraire les données ligne par ligne dans l'ordre exact du tableau original.
- Les colonnes originales seront préservées dans le résultat final pour référence dans la liste names_of_columns_used_for_extraction.
"""

def process_geotechnical_data(
    dataframe: pd.DataFrame, 
    llm_model: str = 'gemini-2.0-flash'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Processes a DataFrame using an LLM to extract and structure geotechnical data
    into two separate DataFrames, preserving original columns.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (main_formation_data_with_originals, supplementary_data_with_originals)
    """

    def create_agent(model_class: type[GeotechnicalBaseModel], 
                    llm_model: str ) -> Agent:
        """Create an agent for the given model class"""
        return Agent(llm_model, output_type=model_class, sys_prompt=model_class.sys_prompt)
    
    main_agent = create_agent(FormationGeotechnique, llm_model)
    extra_agent = create_agent(ValeursSupplementaires, llm_model)
    
    prompt = "Extraire du tableau qui suit les bonnes colonnes et valeurs:\n" + dataframe.to_markdown(index=False)
    
    try:
        main_values = main_agent.run_sync(prompt)
        extra_values = extra_agent.run_sync(prompt)
        
        # Pass original dataframe to preserve columns
        main_df = main_values.to_dataframe(original_df=dataframe)
        extra_df = extra_values.to_dataframe(original_df=dataframe)
        
        return main_df, extra_df
        
    except Exception as e:
        st.warning(f"Error processing geotechnical data: {e}")
        # Return empty DataFrames with correct columns on error
        main_cols = list(FormationGeotechnique.model_fields.keys())[1:]
        extra_cols = list(ValeursSupplementaires.model_fields.keys())[1:]
                
        main_df = pd.DataFrame(columns=main_cols)
        extra_df = pd.DataFrame(columns=extra_cols)
        return main_df, extra_df


class Basique(BaseModel):
    names_of_columns_used_for_extraction: conlist( str, min_length=3, max_length=3 )

    sys_prompt: ClassVar[str] = """
    À partir du tableau, identifiez les 3 colonnes qui correspondent le mieux à :
    
    - **profondeur_de_base** : colonne numérique représentant une profondeur (souvent en mètres).
    - **classes_de_sol** : colonne textuelle décrivant des types de sols comme 'argile', 'sable', 'limon', etc.
    - **pression_limite_retenu** : colonne numérique indiquant une pression limite, souvent notée `pl` ou `pi`.
    
    Retournez uniquement :
    ```json
    {
      "names_of_columns_used_for_extraction": [
        "NomColonneProfondeur",
        "NomColonneSol",
        "NomColonnePression"
      ]
    }"""

def process_geotechnical_data(
    dataframe: pd.DataFrame, 
    llm_model: str = 'gemini-2.0-flash'
) -> pd.DataFrame:
    def create_agent(model_class: type[BaseModel], 
                    llm_model: str ) -> Agent:
        """Create an agent for the given model class"""
        return Agent(llm_model, output_type=model_class, sys_prompt=model_class.sys_prompt)
    
    main_agent = create_agent(Basique, llm_model)

    prompt = "Extraire du tableau qui suit les bonnes colonnes:\n" + dataframe.to_markdown(index=False)
    
    main_values = main_agent.run_sync(prompt)
    
    return main_values


def example_usage():
    """Example demonstrating the enhanced functionality"""
    
    # Sample input data
    sample_data = {
        'Nature des sols': ['Remblais en substitution', 'Remblais', 'Sable graveleux', 'Marno calcaire'],
        'Epaisseur de la formation': ['5.5 m', '11 m', '1.5', ''],
        'Classification sol EC7': ['', 'Argiles et limons', 'Sols intermediaires', 'Marne et calcaire'],
        'Pl* (kPa)': ['', '500', '780', '1300'],
        'qs retenu (kPa)': ['Neglige', '51', '', '142']
    }
    
    df = pd.DataFrame(sample_data)
    print("Input DataFrame:")
    print(df.to_markdown(index=False))
    print("\n" + "="*80 + "\n")
    
    # Process the data
    main_df, extra_df = process_geotechnical_data(df)
    
    print("Main Formation DataFrame (with originals):")
    print(main_df.to_markdown(index=False))
    print("\n" + "="*80 + "\n")
    
    print("Supplementary Values DataFrame (with originals):")
    print(extra_df.to_markdown(index=False))
    
    return main_df, extra_df

    # Expected output structure:
    #"""
    #Main Formation DataFrame would contain:
    #- original_Nature des sols: ['Remblais en substitution', 'Remblais', 'Sable graveleux', 'Marno calcaire']
    #- original_Epaisseur de la formation: ['5.5 m', '11 m', '1.5', '']
    #- original_Classification sol EC7: ['', 'Argiles et limons', 'Sols intermediaires', 'Marne et calcaire']
    #- original_Pl* (kPa): ['', '500', '780', '1300']
    #- original_qs retenu (kPa): ['Neglige', '51', '', '142']
    #- profondeur_de_base: [5.5, 11.0, 12.5, NaN]
    #- classes_de_sol: ['Neutralisé', 'Argile / Limons', 'Sol intermédiaire', 'Marne / Calcaire']
    #- pression_limite_retenu: [NaN, 0.5, 0.78, 1.3]
    #
    #Supplementary Values DataFrame would contain:
    #- All original columns (same as above)
    #- frottement_limite_retenu: [NaN, 51.0, NaN, 142.0]
    #- capacite_de_portance_retenu: [NaN, NaN, NaN, NaN]
    #- modules_pressiometriques: [NaN, NaN, NaN, NaN]
    #- coefficients_rheologiques: [NaN, NaN, NaN, NaN]
    #"""
# ==============================================================================
# 3. SCORING & ANALYSIS
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

def calculate_header_score(headers: List[str], keywords: List[Keyword]) -> Tuple[float, List[str]]:
    """Calculate score based on header matching with keywords."""
    processed_headers = [re.sub(r'[\(\)\[\]\{\}*_]', ' ', h).lower() for h in headers if h]
    
    total_score = 0
    number_of_words = 0
    matched_words_details = set()
    
    for header in processed_headers:
        tokens = set(header.split())
        number_of_words += len(tokens)
        
        for keyword in keywords:
            for token in tokens:
                similarity = fuzz.WRatio(keyword.word.lower(), token)
                if similarity >= keyword.similarity_requirement:
                    total_score += keyword.weight
                    matched_words_details.add(keyword.word)
                    break  # Only count each keyword once per header
    
    multiplier = 3
    normalized_score = multiplier * total_score / log(1 + number_of_words**2) if number_of_words > 0 else 0
    
    return normalized_score, list(matched_words_details)

def calculate_content_score(df: pd.DataFrame, keywords: List[Keyword]) -> Tuple[float, List[str], Dict[Tuple[int, int], List[str]]]:
    """Calculate score based on content matching with keywords."""
    total_weighted_score = 0.0
    matched_target_words = set()
    number_of_options_tested = 0
    match_locations = {}
    
    for r_idx in range(df.shape[0]):
        for c_idx in range(df.shape[1]):
            cell_value = df.iat[r_idx, c_idx]
            if not is_convertible_to_number(cell_value) and str(cell_value).strip():
                cell_content = str(cell_value).lower()
                words_in_cell = cell_content.split()
                number_of_options_tested += len(words_in_cell)
                
                found_in_cell = set()
                for keyword in keywords:
                    if keyword.word not in found_in_cell:
                        similarity = fuzz.partial_ratio(keyword.word.lower(), cell_content)
                        if similarity >= keyword.similarity_requirement:
                            total_weighted_score += keyword.weight
                            matched_target_words.add(keyword.word)
                            found_in_cell.add(keyword.word)
                            match_locations.setdefault((r_idx, c_idx), []).append(keyword.word)
    
    if number_of_options_tested == 0:
        return 0.0, [], {}
    
    multiplier = 3
    normalized_score = multiplier * total_weighted_score / log(1 + number_of_options_tested**2)
    
    return normalized_score, list(matched_target_words), match_locations

def analyze_and_score_tables(dataframe_page_pairs: List[Tuple[pd.DataFrame, int]], keyword_sets: Dict[str, List[Keyword]]) -> List[Dict[str, Any]]:
    """Analyze and score all tables using both header and content analysis."""
    header_keywords = keyword_sets['positive'] + keyword_sets['negative']
    soil_keywords = keyword_sets['soil_types']
    
    results = []
    for i, (df, page) in enumerate(dataframe_page_pairs):
        # Header analysis
        header_score, matched_headers = calculate_header_score(df.columns.tolist(), header_keywords)
        
        # Content analysis (only if table has numeric data or there are few tables)
        if dataframe_has_numeric(df) or len(dataframe_page_pairs) < 200:
            content_score, matched_soil_types, match_details = calculate_content_score(df, soil_keywords)
            final_score = header_score * log(1 + content_score)
        else:
            content_score = 0
            matched_soil_types = []
            match_details = {}
            final_score = header_score
        
        results.append({
            'df_index': i,
            'df': df,
            'page': page,
            'header_score': header_score,
            'bow_score': content_score,
            'final_score': final_score,
            'matched_headers': matched_headers,
            'matched_soil_types': matched_soil_types,
            'match_details': match_details
        })
    
    return sorted(results, key=lambda x: x['final_score'], reverse=True)

# ==============================================================================
# 4. PDF PROCESSING
# ==============================================================================

def parse_page_range(page_range_str: str, num_pages: int) -> List[int]:
    """Parse page range string supporting both commas and ranges (e.g., '1,3,5-8,10')"""
    try:
        page_nums = []
        parts = [p.strip() for p in page_range_str.split(",")]
        
        for part in parts:
            if '-' in part:
                start, end = part.split('-', 1)
                start_num, end_num = int(start.strip()), int(end.strip())
                if start_num <= end_num:
                    page_nums.extend(range(start_num, end_num + 1))
                else:
                    st.warning(f"Range invalide: {part}. Le début doit être inférieur ou égal à la fin.")
            else:
                page_nums.append(int(part))
        
        page_nums = sorted(list(set(page_nums)))
        valid_pages = [p for p in page_nums if 0 <= p < num_pages]
        invalid_pages = [p for p in page_nums if p < 0 or p >= num_pages]
        
        if invalid_pages:
            st.warning(f"Numéros de page invalides ignorés: {invalid_pages} (document contient {num_pages} pages)")
        
        if not valid_pages:
            st.warning("Aucune page valide spécifiée. Utilisation de toutes les pages.")
            return list(range(num_pages))
            
        return valid_pages
        
    except ValueError as e:
        st.warning(f"Format invalide dans la sélection de pages: {e}. Utilisation de toutes les pages.")
        return list(range(num_pages))

def create_filtered_pdf(pdf_bytes: bytes, selected_pages: List[int]) -> str:
    """Create a new PDF with only selected pages and return as temporary file path"""
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    
    with fitz.open(stream=pdf_bytes, filetype="pdf") as original_pdf:
        new_pdf = fitz.open()
        
        for page_num in selected_pages:
            if 0 <= page_num < len(original_pdf):
                new_pdf.insert_pdf(original_pdf, from_page=page_num, to_page=page_num)

        new_pdf.save(temp_pdf.name)
        new_pdf.close()
    
    return temp_pdf.name

def process_with_aryn(file_path: str, use_ocr: bool = True) -> Tuple[Any, str]:
    """Process PDF with Aryn API and return partitioned data"""
    try:
        aryn_api_key = st.secrets.get('ARYN_API_KEY')
        if not aryn_api_key:
            return None, "Clé API Aryn non trouvée dans les secrets Streamlit"
        
        os.environ['ARYN_API_KEY'] = aryn_api_key
        
        with open(file_path, 'rb') as file:
            partitioned_file = partition_file(
                file, 
                aryn_api_key=aryn_api_key, 
                extract_table_structure=True, 
                use_ocr=use_ocr
            )
        
        return partitioned_file, ""
        
    except Exception as e:
        return None, str(e)
