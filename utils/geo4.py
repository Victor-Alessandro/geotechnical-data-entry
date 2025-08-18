import os
import numpy as np
import streamlit as st
import pandas as pd
import re
import io
import tempfile
import json
import spacy

from aryn_sdk.partition import partition_file
from chunkr_ai import Chunkr
from bs4 import BeautifulSoup

from spacy.matcher import Matcher
from PyPDF2 import PdfReader, PdfWriter
from typing import List, Dict, Any, Tuple, Optional, Literal, ClassVar, NamedTuple 

from pydantic import BaseModel, model_validator, conlist
from pydantic_ai import Agent

# abstract nonsense
from abc import ABC

from collections import namedtuple
from math import log
from model2vec import StaticModel
from sklearn.metrics.pairwise import cosine_similarity

# I'll need to put huggingface token on secrets and load it as an os variable, because of sentence transformers
class ExtractionConfig(NamedTuple):
    matcher_id: str
    patterns: List[List[Dict[str, Any]]]
    query: str  

# maybe negate mechanique and use fuzzy2 for technique
EXTRACTION_CONFIGS = [
    ExtractionConfig(
        "Technique des Pieux", 
        [[{"LOWER": "pieux"}], [{"LOWER": {"FUZZY1": "technique"}}], [{"LOWER": {"FUZZY2": "longueur"}}]],
        "Quelle technique devrait être utilisée pour les pieux ? Vissé moulé, Battu Moulé, Foré Tarière Creuse..."
    ),
    ExtractionConfig(
        "Terrain Naturel", 
        [[{"LOWER": "ngf"}], [{"LOWER": {"FUZZY2": "terrain naturel"}}]],
        "Quelle est l'hauteur en NGF du terrain naturel ? Cela veut-dire le sol en place dans son état initial."
    ),
    ExtractionConfig(
        "Profondeur d'Ancrage", 
        [[{"LOWER": {"FUZZY2": "ancrage"}}], [{"LOWER": {"REGEX": "^ancr"}}], [{"LOWER": "profondeur"}]],
        "Quelles sont les recommendations pour la profondeur d'ancrage des pieux ?"
    ),
    ExtractionConfig(
        "Sismicité", 
        [[{"LOWER": {"FUZZY2": "sismique"}}], [{"LOWER": {"FUZZY2": "sismicité"}}]],
        "Dans quelle zone de sismicité le site se situe-t-il ?"
    ),
    ExtractionConfig(
        "Type de Bâtiment", 
        [[{"LOWER": {"FUZZY2": "construction"}}], [{"LOWER": {"FUZZY2": "batiment"}}]],
        "Quel type de bâtiment est construit ? Habitation, lieux de réunion, commerces, réservoirs..."
    ),
    ExtractionConfig(
        "Niveau",
        [[{"TEXT": {"REGEX": "^[Rr][\\+\\-._ ,]*[0-9olisgb]+$"}}], [{"LOWER": "niveau"}]],
        "Quel est le niveau de référence ou l'étage du bâtiment ? Comme R+1, R-1, niveau 0." # Added a relevant query
    ),
    ExtractionConfig(
        "Localisation", 
        [[{"LOWER": {"FUZZY1": "commune"}}], [{"LOWER": {"FUZZY1": "ville"}}], [{"LOWER": {"FUZZY1": "chantier"}}], [{"LOWER": {"FUZZY1": "locale"}}]],
        "Dans quelle commune le bâtiment est-il construit ?"
    ),
    ExtractionConfig(
        "Agressivité du Béton",
        [[{"TEXT": {"REGEX": "^[xX][acdfsACDFS][1-4oli]$"}}], [{"LOWER": {"FUZZY2": "agressivité"}}, {"LOWER": {"FUZZY2": "béton"}}], [{"LOWER": {"FUZZY1": "beton"}}]],
        "Selon l'agressivité du sol quel type de béton est recommandé ? Par exemple C30/37 XA1."
    ),
]

def get_spacy_nlp() -> Any:
    """Loads the spaCy model, downloading if necessary."""
    model_name = 'fr_core_news_sm'
    try:
        nlp = spacy.load(model_name)
    except OSError:
        st.info(f"Downloading spaCy model '{model_name}'... Please wait.")
        try:
            spacy.cli.download(model_name)
            nlp = spacy.load(model_name)
        except Exception as e:
            st.error(f"Failed to download or load spaCy model: {e}")
            return None
    if 'parser' not in nlp.pipe_names and 'sentencizer' not in nlp.pipe_names:
        try:
            nlp.add_pipe('sentencizer', first=True)
        except Exception as e:
            st.warning(f"Could not add sentencizer. Sentence segmentation might be suboptimal. Error: {e}")
    return nlp

def extract_items_from_elements(json_data: Dict[str, Any], nlp: spacy.Language) -> List[Tuple[str, str, int]]:
    """
    Extracts sentences from JSON using efficient batch processing.
    
    """
    items: List[Tuple[str, str, int]] = []
    elements = json_data.get('elements', [])
    if not isinstance(elements, list):
        st.warning("OCR result format is invalid: 'elements' is not a list.")
        return []

    texts_to_process = []
    metadata = []
    for element in elements:
        if isinstance(element, dict) and element.get('text_representation'):
            typ = element.get('type', 'Unknown')

            # maybe keep table
            if typ not in ('Image', 'Page-header', 'Page-footer', 'Table'):
                text = element['text_representation'].strip()
                if text:
                    page = element.get('properties', {}).get('page_number', 0)
                    texts_to_process.append(text)
                    metadata.append({'type': typ, 'page': page})

    for doc, meta in zip(nlp.pipe(texts_to_process), metadata):
        for sent in doc.sents:
            sentence = sent.text.strip()
            if sentence:
                items.append((sentence, meta['type'], meta['page']))
    return items


def extract_phrases(items: List[Tuple[str, str, int]],
                    configs: List[ExtractionConfig],
                    nlp: spacy.Language) -> Dict[str, List[Dict[str, Any]]]:
    """
    For each config, return full item strings whose sentences match any pattern.
    """
    matcher = Matcher(nlp.vocab)
    for config in configs:
        matcher.add(config.matcher_id, config.patterns)
    
    categorized: Dict[str, List[Dict[str, Any]]] = {cfg.matcher_id: [] for cfg in configs}
    seen: Dict[str, set] = {cfg.matcher_id: set() for cfg in configs}

    texts = [sent for sent, _, _ in items]
    disable_pipes = [pipe for pipe in nlp.pipe_names if pipe not in ('sentencizer',)]

    for idx, doc in enumerate(nlp.pipe(texts, disable=disable_pipes)):
        matches = matcher(doc)
        if matches:
            original_sentence, _, page = items[idx]
            
            highlighted_sentence = original_sentence
            # Get character indices for all matches and sort them in reverse
            # to avoid messing up indices during insertion.
            spans = sorted([match[1:] for match in matches], key=lambda s: doc[s[0]:s[1]].start_char, reverse=True)
            
            for start, end in spans:
                span = doc[start:end]
                start_char, end_char = span.start_char, span.end_char
                highlighted_sentence = (
                    highlighted_sentence[:start_char] 
                    + f"**{span.text}**" 
                    + highlighted_sentence[end_char:]
                )

            for match_id, _, _ in matches:
                label = nlp.vocab.strings[match_id]
                if (original_sentence, page) not in seen[label]:
                    seen[label].add((original_sentence, page))
                    categorized[label].append({
                        'sentence': original_sentence,
                        'highlighted': highlighted_sentence,
                        'page': page,
                    })

    return categorized

@st.cache_resource
def load_embedder():
    """Loads the Model2Vec model and caches it."""

    model_name = os.getenv("MODEL2VEC_MODEL", "minishlab/potion-multilingual-128M")

    try:
        return StaticModel.from_pretrained(model_name)
    except Exception as e:
        # Fallback to a smaller model if the preferred one fails
        print(f"Failed to load {model_name}, falling back to base model: {e}")
        return StaticModel.from_pretrained("minishlab/M2V_base_output")

#does the static model need this?  @st.cache_data(hash_funcs={SentenceTransformer: id})
def get_reranked_results(embedder: StaticModel, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """
    Re-ranks results using Model2Vec embeddings and cosine similarity.
    
    Args:
        embedder: The Model2Vec model
        results: List of dictionaries containing text results
        query: The query string to compare against
        
    Returns:
        List of results sorted by semantic similarity to the query
    """
    if not results:
        return []
    
    texts = [result.get('text', '') for result in results]
    
    query_embedding = embedder.encode([query])
    text_embeddings = embedder.encode(texts)
    
    # NOT IDEAL, THE QUERY IS JUST THAT A QUERY. WHAT I NEED IS AN ANSWER, NOT TEXT SIMILAR TO THE QUERY 
    similarities = cosine_similarity(query_embedding, text_embeddings)[0]
    
    for i, result in enumerate(results):
        result['similarity_score'] = float(similarities[i])
    
    return sorted(results, key=lambda x: x['similarity_score'], reverse=True)

def run_extraction_pipeline(items, extraction_config, nlp: spacy.Language):
    """
    Executes the full pipeline: element extraction, keyword matching, 
    semantic re-ranking, and displays the final results.
    """
    embedder = load_embedder()
   
    categorized_phrases = extract_phrases(items, extraction_config, nlp)
    reranked_results: Dict[str, List[Dict[str, Any]]] = {}
    
    query_map = {config.matcher_id: config.query for config in EXTRACTION_CONFIGS}
    
    for category_name, results in categorized_phrases.items():
        if results:
            query = query_map[category_name]
            reranked_results[category_name] = get_reranked_results(embedder, results, query)
        else:
            reranked_results[category_name] = []
    
    return reranked_results

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

Keyword = namedtuple('Keyword', ['word', 'weight', 'patterns'])

# I might abstract this to work on any namedtuple and then curry to define create_keywords
@st.cache_data
def create_keywords(keyword_tuples: List[Tuple[str, float, List[List[Dict[str, Any]]]]]) -> List[Keyword]:
    """Convert list of tuples to list of Keyword namedtuples with spaCy patterns."""
    return [Keyword(word, weight, patterns) for word, weight, patterns in keyword_tuples]

def create_spacy_patterns_for_keyword(word: str, fuzzy_level: int = 0) -> List[List[Dict[str, Any]]]:
    """
    Create spaCy matcher patterns for a keyword with different fuzzy matching levels.
    
    Args:
        word: The keyword to match
        fuzzy_level: 0=exact, 1=FUZZY1, 2=FUZZY2... 
    """
    word_lower = word.lower()
    
    if fuzzy_level == 0:
        return [[{"LOWER": word_lower}]]
    elif fuzzy_level < 10:
        # FUZZY - fuzzy matching, FUZZY1 -> levehstein distance 1
        fuzzy_flexibility = "FUZZY" + str(fuzzy_level)
        return [[{"LOWER": {fuzzy_flexibility : word_lower}}]]
    else:
        # Fallback to exact match        
        escaped_word = re.escape(word_lower)
        return [[{"LOWER": {"REGEX": f"^{escaped_word}"}}]]

@st.cache_data
def get_keyword_sets() -> Dict[str, List[Keyword]]:
    """Define all keyword sets used for analysis with spaCy patterns."""
    # Define keywords with fuzzy levels (0=exact, 1=FUZZY1, 2=FUZZY2...)
    positive_keywords = [
        ('formation', 1, 1), ('nature', 1, 1), ('sol', 1, 0), 
        ('horizon', 1, 1), ('couche', 1, 2), ('faciès', 1, 2), 
        ('profondeur', 1, 1), ('prof', 1, 1), ('épaisseur', 1, 2), 
        ('mpa', 1, 0), ('pl', 2.8, 0), ('pi', 2.8, 0), 
        ('pression', 1, 1), ('kp', 2.5, 0), ('courbe', 1.0, 1),
        ('alfa', 2.0, 0), ('α', 2.0, 0),
        ('em', 0.5, 0), ('terrain', 1, 1)
    ]
    
    negative_keywords = [
        ('avancement', -15, 1), ('vitesse', -15, 1), ('injection', -15, 1),
        ('rotation', -15, 1), ('fluage', -15, 1), ('sondage', -10, 1)
    ]
    
    soil_type_keywords = [
        ('sable', 1, 0), ('alluvions', 1, 0), ('remblais', 1, 0), 
        ('tuffeau', 1, 0), ('craie', 1, 0), ('argile', 1, 0), 
        ('limons', 1, 0), ('marne', 1, 0), ('graves', 1, 0), 
        ('calcaire', 1, 0), ('roche', 1, 0),
        ('retenue', 3.0, 1), ('max', 1, 0), ('min', 1, 0)
    ]
    
    positive_tuples = [(word, weight, create_spacy_patterns_for_keyword(word, fuzzy_level)) 
                      for word, weight, fuzzy_level in positive_keywords]
    negative_tuples = [(word, weight, create_spacy_patterns_for_keyword(word, fuzzy_level)) 
                      for word, weight, fuzzy_level in negative_keywords]
    soil_type_tuples = [(word, weight, create_spacy_patterns_for_keyword(word, fuzzy_level)) 
                       for word, weight, fuzzy_level in soil_type_keywords]
    
    return {
        'positive': create_keywords(positive_tuples),
        'negative': create_keywords(negative_tuples),
        'soil_types': create_keywords(soil_type_tuples)
    }

# ==============================================================================
# 2. DATA PARSING & PREPARATION
# ==============================================================================

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
                # , and the courbe case is redacted...
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
        return False # return False for cases that don't match any condition
    
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
    llm_model: str = 'gemini-2.5-pro'
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


def clean_column_names(df):
    """
    Simplified version that keeps only the first unique non-duplicate part
    """
    cleaned_columns = []
    
    for col in df.columns:
        # Split by common delimiters
        parts = re.split(r'[|\\/?&]+', str(col))
        
        # Clean and find first non-empty, non-duplicate part
        seen_parts = set()
        final_part = None
        
        for part in parts:
            # Clean the part
            cleaned = re.sub(r'[^\w\s()\-*.,]', '', part.strip())
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            if cleaned and cleaned.lower() not in seen_parts:
                if final_part is None:
                    final_part = cleaned
                seen_parts.add(cleaned.lower())
        
        if final_part:
            cleaned_columns.append(final_part)
        else:
            cleaned_columns.append(f"Column_{len(cleaned_columns)}")
    
    df_cleaned = df.copy()
    df_cleaned.columns = cleaned_columns
    return df_cleaned

class Basique(BaseModel):
    names_of_columns_used_for_extraction: conlist( str, min_length=3, max_length=3 )

    sys_prompt: ClassVar[str] = """
    À partir du tableau, identifiez les 3 colonnes qui correspondent le mieux à :
    
    - **profondeur_de_base** : colonne numérique représentant une profondeur (souvent en mètres).
    - **classes_de_sol** : colonne textuelle décrivant des types de sols comme 'argile', 'sable', 'limon', etc.
    - **pression_limite_retenu** : colonne numérique indiquant une pression limite, souvent nomée `pl` ou `pi`.
    
    Retournez uniquement :
    ```json
    {
      "names_of_columns_used_for_extraction": [
        "NomColonneProfondeur",
        "NomColonneSol",
        "NomColonnePressionLimite"
      ]
    }"""



def placeholder_process_geotechnical_data(
    dataframe: pd.DataFrame, 
    llm_model: str = 'gemini-2.5-flash'
) -> pd.DataFrame:
    def create_agent(model_class: type[BaseModel], 
                    llm_model: str ) -> Agent:
        """Create an agent for the given model class"""
        return Agent(llm_model, output_type=model_class, system_prompt=model_class.sys_prompt)
    
    main_agent = create_agent(Basique, llm_model)

    prompt = "Extraire du tableau qui suit les bonnes colonnes:\n" + dataframe.to_markdown(index=False)
    
    main_values = main_agent.run_sync(prompt)
    
    return main_values

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

def calculate_header_score_with_spacy(headers: List[str], keywords: List[Keyword], nlp: spacy.Language) -> Tuple[float, List[str]]:
    """Calculate score based on header matching using spaCy Matcher."""
    if not headers:
        return 0.0, []
    
    # Create a temporary matcher for this operation
    matcher = Matcher(nlp.vocab)
    
    # Add patterns for each keyword
    for i, keyword in enumerate(keywords):
        matcher_id = f"keyword_{i}_{keyword.word}"
        matcher.add(matcher_id, keyword.patterns)
    
    processed_headers = [re.sub(r'[\(\)\[\]\{\}*_]', ' ', h) for h in headers if h]
    header_text = " ".join(processed_headers).lower()
    
    if not header_text.strip():
        return 0.0, []
    
    doc = nlp(header_text)
    matches = matcher(doc)
    
    total_score = 0
    matched_words = set()
    word_counts = {}
    
    # Count matches and calculate scores
    for match_id, start, end in matches:
        # Extract keyword info from the match_id
        match_label = nlp.vocab.strings[match_id]
        
        # Find the corresponding keyword
        for keyword in keywords:
            if keyword.word in match_label:
                if keyword.word not in word_counts:
                    word_counts[keyword.word] = 0
                word_counts[keyword.word] += 1
                matched_words.add(keyword.word)
                
                # Only count each keyword once per header analysis
                if word_counts[keyword.word] == 1:
                    total_score += keyword.weight
                break
    
    # Normalize by number of words (with logarithmic scaling)
    number_of_words = len(doc)
    multiplier = 3
    normalized_score = multiplier * total_score / log(1 + number_of_words**2) if number_of_words > 0 else 0
    
    return normalized_score, list(matched_words)

def calculate_content_score_with_spacy(df: pd.DataFrame, keywords: List[Keyword], nlp: spacy.Language) -> Tuple[float, List[str], Dict[Tuple[int, int], List[str]]]:
    """Calculate score based on content matching using spaCy Matcher."""
    if df.empty:
        return 0.0, [], {}
    
    # Create a temporary matcher for this operation
    matcher = Matcher(nlp.vocab)
    
    # Add patterns for each keyword
    for i, keyword in enumerate(keywords):
        matcher_id = f"content_keyword_{i}_{keyword.word}"
        matcher.add(matcher_id, keyword.patterns)
    
    total_weighted_score = 0.0
    matched_target_words = set()
    number_of_options_tested = 0
    match_locations = {}
    
    # Collect all non-numeric text content
    text_cells = []
    cell_positions = []
    
    for r_idx in range(df.shape[0]):
        for c_idx in range(df.shape[1]):
            cell_value = df.iat[r_idx, c_idx]
            if not is_convertible_to_number(cell_value) and str(cell_value).strip():
                cell_content = str(cell_value).lower()
                text_cells.append(cell_content)
                cell_positions.append((r_idx, c_idx))
    
    if not text_cells:
        return 0.0, [], {}
    
    for idx, doc in enumerate(nlp.pipe(text_cells)):
        r_idx, c_idx = cell_positions[idx]
        matches = matcher(doc)
        number_of_options_tested += len(doc)
        
        cell_matched_words = set()
        
        for match_id, start, end in matches:
            match_label = nlp.vocab.strings[match_id]
            
            # Find the corresponding keyword
            for keyword in keywords:
                if keyword.word in match_label and keyword.word not in cell_matched_words:
                    # Only count each keyword once per cell
                    cell_matched_words.add(keyword.word)
                    matched_target_words.add(keyword.word)
                    total_weighted_score += keyword.weight
                    match_locations.setdefault((r_idx, c_idx), []).append(keyword.word)
                    break
    
    if number_of_options_tested == 0:
        return 0.0, [], {}
    
    multiplier = 3
    normalized_score = multiplier * total_weighted_score / log(1 + number_of_options_tested**2)
    
    return normalized_score, list(matched_target_words), match_locations

def analyze_and_score_tables(dataframe_page_pairs: List[Tuple[pd.DataFrame, int]], keyword_sets: Dict[str, List[Keyword]]) -> List[Dict[str, Any]]:
    """Analyze and score all tables using spaCy Matcher for both header and content analysis."""
    # Get the nlp instance
    nlp = get_spacy_nlp()
    if nlp is None:
        st.error("Could not load spaCy model. Table scoring will fail.")
        return []
    
    header_keywords = keyword_sets['positive'] + keyword_sets['negative']
    soil_keywords = keyword_sets['soil_types']
    
    results = []
    for i, (df, page) in enumerate(dataframe_page_pairs):
        header_score, matched_headers = calculate_header_score_with_spacy(df.columns.tolist(), header_keywords, nlp)
        
        if dataframe_has_numeric(df) or len(dataframe_page_pairs) < 200:
            content_score, matched_soil_types, match_details = calculate_content_score_with_spacy(df, soil_keywords, nlp)
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

# filtering as in the pdf is considered a list of pages which we filter according to user inputs
def create_filtered_pdf(pdf_bytes: bytes, selected_pages: List[int]) -> str:
    """Create a new PDF with only selected pages and return the temporary file path."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    writer = PdfWriter()

    for page_num in selected_pages:
        if 0 <= page_num < len(reader.pages):
            writer.add_page(reader.pages[page_num])

    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    with open(temp_pdf.name, "wb") as f:
        writer.write(f)

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



def convert_to_aryn_format(input_data):
    """
    Converts a custom JSON format to the Aryn.ai docparse output structure.
    This function processes document segments from the input data and converts them
    into the Aryn.ai 'elements' format, as described in the Aryn documentation.
    It correctly calculates proportional bounding boxes and handles different
    element types like Text, Image, and Table with proper Aryn.ai structure.
    The final output is a dictionary with a single key, 'elements', which
    contains a list of the converted document elements.
    Args:
        input_data (dict): The input data parsed from the source JSON,
                             containing 'chunks' and 'segments'.
    Returns:
        dict: A dictionary structured as { "elements": [...] } that can be
              serialized into the target JSON format.
    """
    # Mapping from the input segment_type to the Aryn.ai element type
    # based on the actual Aryn.ai output format
    type_mapping = {
        "Picture": "Image",
        "Paragraph": "Text", 
        "Table": "table",  # lowercase as per Aryn.ai format
        "Header": "Section-header",
        "Footer": "Page-footer",
        "List": "List-item",
        # Default for unknown types will be "Text"
    }
    
    def parse_html_table_to_aryn_format(html_content):
        """Parse HTML table content into Aryn.ai table structure"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            table = soup.find('table')
            
            if not table:
                return None
                
            cells = []
            column_headers = []
            rows = table.find_all('tr')
            
            for row_idx, row in enumerate(rows):
                cols = row.find_all(['th', 'td'])
                for col_idx, col in enumerate(cols):
                    # Handle colspan and rowspan
                    colspan = int(col.get('colspan', 1))
                    rowspan = int(col.get('rowspan', 1))
                    
                    # Create cell data
                    cell_data = {
                        'content': col.get_text(strip=True),
                        'rows': list(range(row_idx, row_idx + rowspan)),
                        'cols': list(range(col_idx, col_idx + colspan)),
                        'is_header': col.name == 'th' or row_idx == 0,
                        'bbox': {
                            'x1': 0.0, 'y1': 0.0, 'x2': 0.0, 'y2': 0.0  # Placeholder
                        },
                        'properties': {}
                    }
                    cells.append(cell_data)
                    
                    # Build column headers
                    if row_idx < 2:  # Typically first two rows are headers
                        for c in range(col_idx, col_idx + colspan):
                            if c >= len(column_headers):
                                column_headers.extend([''] * (c - len(column_headers) + 1))
                            if column_headers[c]:
                                column_headers[c] += f" | {col.get_text(strip=True)}"
                            else:
                                column_headers[c] = col.get_text(strip=True)
            
            return {
                'cells': cells,
                'caption': None,
                'num_rows': len(rows),
                'num_cols': max(len(row.find_all(['th', 'td'])) for row in rows) if rows else 0,
                'column_headers': column_headers
            }
        except ImportError:
            return None
    
    processed_elements = []
    for chunk in input_data.get('chunks', []):
        for segment in chunk.get('segments', []):

            # 1. Determine Element Type
            input_type = segment.get('segment_type')
            aryn_type = type_mapping.get(input_type, "Text")

            # 2. Calculate Proportional Bounding Box
            # The Aryn format requires proportional coordinates (0.0 to 1.0).
            bbox_in = segment.get('bbox')
            page_w = segment.get('page_width')
            page_h = segment.get('page_height')
            output_bbox = [0.0, 0.0, 0.0, 0.0]
            if bbox_in and page_w and page_h and page_w > 0 and page_h > 0:
                left = bbox_in.get('left', 0)
                top = bbox_in.get('top', 0)
                width = bbox_in.get('width', 0)
                height = bbox_in.get('height', 0)
                output_bbox = [
                    left / page_w,
                    top / page_h,
                    (left + width) / page_w,
                    (top + height) / page_h
                ]

            # 3. Assemble Properties
            properties = {}
            if segment.get('confidence') is not None:
                properties['score'] = segment.get('confidence')
            if segment.get('page_number') is not None:
                properties['page_number'] = segment.get('page_number')

            # 4. Get Text Representation
            text_representation = segment.get('content', '')
            
            if aryn_type == "table" and segment.get('text'):
                text_representation = segment.get('text', text_representation)

            # 5. Assemble Base Element Dictionary
            aryn_element = {
                "type": aryn_type,
                "bbox": output_bbox,
                "properties": properties,
                "text_representation": text_representation
            }
            # 6. Handle Type-Specific Fields as per the Aryn.ai format
            if aryn_type == "Image":
                if bbox_in:
                    # Add image-specific properties
                    properties['image_size'] = [int(bbox_in.get('width', 0)), int(bbox_in.get('height', 0))]
                    properties['image_mode'] = None
                    properties['image_format'] = None
                # Add top-level binary_representation key for Images.
                aryn_element['binary_representation'] = None
                
            elif aryn_type == "table":
                table_content = segment.get('content', '')
                
                if table_content:
                    properties['title'] = None
                    # Try to parse the HTML table structure
                    parsed_table = parse_html_table_to_aryn_format(table_content)
                    if parsed_table:
                        properties['columns'] = parsed_table['num_cols']
                        properties['rows'] = parsed_table['num_rows']
                        aryn_element['table'] = parsed_table
                    else:
                        # Fallback for when HTML parsing fails
                        properties['columns'] = 0
                        properties['rows'] = 0
                        aryn_element['table'] = {
                            'cells': [],
                            'caption': None,
                            'num_rows': 0,
                            'num_cols': 0,
                            'column_headers': []
                        }
                aryn_element['text_representation'] = None
            processed_elements.append(aryn_element)
    return {"elements": processed_elements}

def process_with_chunkr(file_path: str, use_ocr: bool = True) -> Tuple[Any, str]:
    """Process PDF with Chunkr API and return extracted JSON data"""
    try:
        api_key = st.secrets.get("CHUNKR_API_KEY")
        if not api_key:
            return None, "Clé API Chunkr non trouvée dans les secrets Streamlit"

        with open(file_path, 'rb') as f:
            pdf_bytes_io = io.BytesIO(f.read())

        chunkr = Chunkr(api_key=api_key)
        try:
            task = chunkr.upload(pdf_bytes_io)

            raw = task.output.json()         # this is a JSON string
            if isinstance(raw, str):
                json_data = json.loads(raw)  # now a dict
            else:
                json_data = raw              # just in case it was already parsed

            aryn_payload = convert_to_aryn_format(json_data)

            return aryn_payload, ""
        finally:
            chunkr.close()

    except Exception as e:
        return None, str(e)
