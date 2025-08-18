from docxtpl import DocxTemplate, InlineImage
from io import BytesIO
from docx.shared import Mm
from hashlib import md5
from itertools import chain
from pdf2image import convert_from_bytes, convert_from_path

import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import contextily as ctx
import numpy as np
import io
import math
import cv2
import json
import os
import tempfile
import shutil
import requests
import gc


TECHNIQUES = {
    "Forés Tarière Creuse": {
        "description_technique": "pieux forés tarière creuse, utilisant une foreuse munie des appareils d'enregistrement continu des paramètres de forage et de bétonnage",
        "norme_associe": "norme NFP 94-262, les pieux sont notés FTC, classe 2 et catégorie 6."
    },
    "Vissés Moulé": {
        "description_technique": "pieux vissés moulé, avec mise en place par vissage",
        "norme_associe": "norme NFP 94-262, les pieux sont notés VM."
    },
    "Battus Moulé": {
        "description_technique": "pieux battus moulé, avec mise en place par battage",
        "norme_associe": "norme NFP 94-262, les pieux sont notés BM."
    }
}

BETON_TYPES = {
    "Béton type C30/37 - Agressivité XC1": "Étant donnée le milieu XC1 sec ou humide en permanence (liant ≥ 260–280 kg/m³), conduisant à l'utilisation du béton C30/37 selon la norme NF EN 206-1.",
    "Béton type C30/37 - Agressivité XC2": "Étant donnée le milieu humide, rarement sec (liant ≥ 280 kg/m³), conduisant à l'utilisation du béton C30/37 selon la norme NF EN 206-1.",
    "Béton type C30/37 - Agressivité XC3": "Étant donnée le milieu humidité modérée (liant ≥ 300 kg/m³), conduisant à l'utilisation du béton C30/37 selon la norme NF EN 206-1.",
    "Béton type C30/37 - Agressivité XC4": "Étant donnée le milieu cycles d'humidité/séchage (liant ≥ 315 kg/m³), conduisant à l'utilisation du béton C30/37 selon la norme NF EN 206-1.",
    "Béton type C30/37 - Agressivité XF2": "Étant donnée le milieu gel faible ou modéré avec sels de déverglaçage ou avec eau stagnante (liant ≥ 315 kg/m³), conduisant à l'utilisation du béton C30/37 selon la norme NF EN 206-1.",
    "Béton type C30/37 - Agressivité XF3": "Étant donnée le milieu gel sévère avec humidité importante (liant ≥ 340 kg/m³), conduisant à l'utilisation du béton C30/37 selon la norme NF EN 206-1.",
    "Béton type C30/37 - Agressivité XF4": "Étant donnée le milieu gel sévère avec sels de déverglaçage (liant ≥ 340 kg/m³), conduisant à l'utilisation du béton C30/37 selon la norme NF EN 206-1.",
    "Béton type C30/37 - Agressivité XD1": "Étant donnée le milieu chlorures non marins sans humidité constante (liant ≥ 315 kg/m³), conduisant à l'utilisation du béton C30/37 selon la norme NF EN 206-1.",
    "Béton type C30/37 - Agressivité XD3": "Étant donnée le milieu chlorures non marins avec cycles humide/sec (liant ≥ 350 kg/m³), conduisant à l'utilisation du béton C30/37 selon la norme NF EN 206-1.",
    "Béton type C30/37 - Agressivité XS1": "Étant donnée le milieu air contenant des chlorures marins (liant ≥ 330 kg/m³), conduisant à l'utilisation du béton C30/37 selon la norme NF EN 206-1.",
    "Béton type C30/37 - Agressivité XA1": "Étant donnée le milieu agression chimique faible (liant ≥ 330 kg/m³), conduisant à l'utilisation du béton C30/37 selon la norme NF EN 206-1.",
    "Béton type C35/45 - Agressivité XS2": "Étant donnée le milieu immersion partielle eau de mer (liant ≥ 350 kg/m³), conduisant à l'utilisation du béton C35/45 selon la norme NF EN 206-1.",
    "Béton type C35/45 - Agressivité XS3": "Étant donnée le milieu immersion totale eau de mer / embruns fréquents (liant ≥ 350 kg/m³), conduisant à l'utilisation du béton C35/45 selon la norme NF EN 206-1.",
    "Béton type C35/45 - Agressivité XA2": "Étant donnée le milieu agression chimique modérée (liant ≥ 350 kg/m³), conduisant à l'utilisation du béton C35/45 selon la norme NF EN 206-1.",
    "Béton type C35/45 - Agressivité XD2": "Étant donnée le milieu chlorures non marins en milieu humide (liant ≥ 350 kg/m³), conduisant à l'utilisation du béton C35/45 selon la norme NF EN 206-1.",
    "Béton type C40/45 - Agressivité XA3": "Étant donnée le milieu agression chimique forte (liant ≥ 360 kg/m³), conduisant à l'utilisation du béton C40/45 selon la norme NF EN 206-1.",
    "Béton type C25/30 - Agressivité XF1": "Étant donnée le milieu gel faible ou modéré sans sels de déverglaçage (liant ≥ 300 kg/m³), conduisant à l'utilisation du béton C25/30 selon la norme NF EN 206-1.",
    "Béton type C16/20 - Agressivité XA0": "Étant donnée le milieu non agressif, sans risque de corrosion ou exposition chimique significative. (liant ≥ 150–260 kg/m³), conduisant à l'utilisation du béton C16/20 selon la norme NF EN 206-1.",
}

ACIER_TYPES = {
    "Acier HA - Fe E500": "Les pieux seront munis d'aciers type HA Fe E500 conformément aux normes en vigueur."
}

DIAMETER_OPTIONS = ["Un seul diamètre", "Plusieurs diamètres"]

# --- Image Production ---

# this name is bad, it has been changed to produce the image and not display
def produce_departement_map(
    gdf_commune: gpd.GeoDataFrame,
    gdf_departement: gpd.GeoDataFrame,
    selected_commune_name: str,
    departement_identifier: str,
    base_crs: str
    ):
  
    fig, ax = plt.subplots(figsize=(10,10))

    ax.set_title(f"Zone de sismicité de {selected_commune_name} (Dép: {departement_identifier})", fontsize=14)
    
    color_map = {
        '1 - Très faible':'gray', '2 - Faible':'gold', '3 - Modérée':'darkorange',
        '4 - Moyenne':'red', '5 - Forte':'#800020'
    }
    gdf_departement['color'] = gdf_departement['Sismicite'].map(color_map).fillna('blue')
    gdf_commune['color'] = gdf_commune['Sismicite'].map(color_map)
    gdf_departement.plot(ax=ax, color=gdf_departement['color'], legend=True, alpha=0.35, edgecolor='grey', linewidth=0.3)
    gdf_commune.plot(ax=ax, color=gdf_commune['color'], alpha=0.9, edgecolor='black', linewidth=0.7)
    ctx.add_basemap(ax, crs=base_crs, source=ctx.providers.OpenStreetMap.Mapnik)
    handles=[]
    commune_cat = gdf_commune['Sismicite'].iloc[0]
    for cat, color in color_map.items():
        if cat == commune_cat:
            patch = mpatches.Patch(facecolor=color, edgecolor='black', linewidth=2.5, label=cat)
        else:
            patch = mpatches.Patch(facecolor=color, edgecolor='k', label=cat)
        handles.append(patch)
    legend = ax.legend(handles=handles, title="Sismicité", loc='upper right', framealpha=0.5)
    legend.get_title().set_fontsize(14)
    for text in legend.get_texts():
        if text.get_text() == commune_cat:
            text.set_fontweight('bold')
            text.set_fontsize(13)
    ax.set_axis_off()
    plt.tight_layout()
    
    # MODIFIED: Removed st.pyplot(fig) to decouple display from generation
    # st.pyplot(fig) 
    
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    
    plt.close(fig)
    gc.collect()
    return buf


def process_commune_api_result(api_res, commune_input):
    """Process the API result for commune search"""
    shapefile_path = "./resources/france_zonage_sismique/France_zonage_sismique.shp"
    try:
        data = gpd.read_file(shapefile_path) 
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier: {e}")
        return None
    
    insee = api_res['code']
    name = api_res.get('nom', commune_input)
    st.session_state.commune = f"{name} ({insee})"
    st.session_state.insee_code = insee
    
    if data is not None:
        match = data[data['insee'] == insee]
        if not match.empty:
            st.toast(f"Commune trouvée: {name} (INSEE: {insee})", icon="✅")
            st.session_state.selected_gdf = match.iloc[[0]]
            st.session_state.selected_name = name
            st.session_state.selected_departement = str(match.iloc[0]['departemen'])
            
            seismic_zones = {
                '1 - Très faible': "Zone 1 (très faible)",
                '2 - Faible': "Zone 2 (faible)",
                '3 - Modérée': "Zone 3 (modérée)",
                '4 - Moyenne': "Zone 4 (moyenne)",
                '5 - Forte': "Zone 5 (forte)"
            }
            if 'Sismicite' in match.columns:
                sismicite_value = match.iloc[0]['Sismicite']
                if sismicite_value in seismic_zones:
                    st.session_state.sismicite = seismic_zones[sismicite_value]
                    st.toast(f"Zone sismique détectée: {st.session_state.sismicite}", icon="ℹ️")
                # Clear map buffer when commune changes to force regeneration if needed
                if 'commune_map_buffer' in st.session_state:
                    st.session_state.commune_map_buffer = None
        else:
            st.toast(f"{name} (INSEE: {insee}) non trouvée dans le fichier local.", icon="⚠️")


@st.cache_data
def get_commune_api(commune_name: str) -> dict | None:
    url = "https://geo.api.gouv.fr/communes"
    params = {"nom": commune_name, "fields": "nom,code,departement", "format": "json"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data:
            return data[0]
    except requests.exceptions.RequestException as e:
        st.toast(f"Erreur lors de la requête API: {e}", icon="⚠️")
    except Exception as e:
        st.toast(f"Une erreur inattendue est survenue lors de l'appel API: {e}", icon="🛑")
    return None

def handle_commune_input():
    """Handle commune input with API search"""
    previous_commune_val = st.session_state.commune.split(" (")[0] if "(" in st.session_state.commune else st.session_state.commune
    commune_input = st.text_input("Commune", value=previous_commune_val)
    
    if commune_input and commune_input != previous_commune_val:
        with st.spinner("Recherche de la commune..."):
            api_res = get_commune_api(commune_input)
            if api_res and api_res.get('code'):
                process_commune_api_result(api_res, commune_input)
            else:
                st.toast(f"Aucun résultat API pour '{commune_input}'.", icon="⚠️")
    elif not commune_input and previous_commune_val: # handles case where user deletes input
        st.session_state.commune = ""
        st.session_state.selected_gdf = None
        st.session_state.sismicite = st.session_state.get("defaults", {}).get("sismicite", "Zone 1 (très faible)")
        if 'commune_map_buffer' in st.session_state:
            st.session_state.commune_map_buffer = None
        if 'highlighted_image' in st.session_state:
            st.session_state.highlighted_image = None


def get_coords_to_highlight( i, j): # i,j are 1-based
    top_left_coords = (101, 80) # original top_left
    cell_width_val = (590 - top_left_coords[0]) // 5
    cell_height_val = (250 - top_left_coords[1]) // 5
    
    x0 = top_left_coords[0] + (j - 1) * cell_width_val
    y0 = top_left_coords[1] + (i - 1) * cell_height_val
    x1 = x0 + cell_width_val
    y1 = y0 + cell_height_val

    return (x0,y0), (x1,y1)

def highlight_coordinates():
    """Generate highlighted seismic norm table image based on current selections"""
    batiment_coord_map = {
        "I": 0, 
        "II avec application possible des PS-MI / CP-MI": 1, 
        "II": 2, 
        "III": 3, 
        "IV": 4
    }
    sismic_coord_map_for_table = { 
        "Zone 1 (très faible)": 0, 
        "Zone 2 (faible)": 1, 
        "Zone 3 (modérée)": 2,
        "Zone 4 (moyenne)": 3, 
        "Zone 5 (forte)": 4
    }
    IMAGE_PATH_NORM = './resources/background.jpg'
           
    i_idx_table = sismic_coord_map_for_table.get(st.session_state.sismicite, -1) 
    j_idx_table = batiment_coord_map.get(st.session_state.type_batiment, -1)
    
    i_coord_for_func = i_idx_table + 1
    j_coord_for_func = j_idx_table + 1
    
    if os.path.exists(IMAGE_PATH_NORM):
        img_norm_table = cv2.imread(IMAGE_PATH_NORM)
        if img_norm_table is not None:
            if i_idx_table != -1 and j_idx_table != -1: # Valid selections
                top_left, lower_right = get_coords_to_highlight(i_coord_for_func, j_coord_for_func)
                highlighted_img_data = img_norm_table.copy()
                cv2.rectangle(highlighted_img_data, top_left, lower_right, (0, 0, 255), 3) 
                return highlighted_img_data # This is the cv2 image array
    return None

def convert_pdf_to_image(pdf_input):
    """
    Convert the first page of a PDF to PNG using pdf2image.

    Parameters:
    - pdf_input: file path (str or Path) or file-like object (e.g., BytesIO)

    Returns:
    - BytesIO: PNG image of the first page
    """
    try:
        if hasattr(pdf_input, "read"):
            # Read from file-like object (e.g., BytesIO)
            pdf_bytes = pdf_input.read()
            images = convert_from_bytes(pdf_bytes, first_page=1, last_page=1)
        else:
            # Read from file path
            images = convert_from_path(pdf_input, first_page=1, last_page=1)

        if not images:
            raise ValueError("PDF contains no pages.")

        img_io = io.BytesIO()
        images[0].save(img_io, format="PNG")
        img_io.seek(0)
        return img_io

    except Exception as e:
        raise RuntimeError(f"Failed to convert PDF to image: {e}")

# --- Excel, Text and Generation ---


def excel_cell_to_coordinates(cell_ref: str) -> tuple[int, int]:
    """
    Convert Excel cell reference (e.g., 'A1', 'AB13') to zero-based coordinates.
    
    Args:
        cell_ref: Excel cell reference like 'A1', 'B2', 'AB13'
    
    Returns:
        tuple: (row, col) zero-based coordinates
    """
    col_str = ''.join(c for c in cell_ref if c.isalpha()).upper()
    row_str = ''.join(c for c in cell_ref if c.isdigit())
    
    col = 0
    for char in col_str:
        col = col * 26 + (ord(char) - ord('A') + 1)
    
    col -= 1 
    row = int(row_str) - 1
    
    return (row, col)


# this function shold be changed if its called too many times, pd.read_excel is called multiple times on the same file. It is an IO operation
# effectively a performance bottleneck.
def extract_excel_zone(excel_file, sheet_name: str, upper_left_cell: str, lower_right_cell: str) -> pd.DataFrame:
    """
    Extract a rectangular zone from an Excel file using cell coordinates.
    
    Args:
        excel_file: Excel file path or file-like object
        sheet_name: Name of the Excel sheet
        upper_left_cell: Upper left cell reference (e.g., 'A1')
        lower_right_cell: Lower right cell reference (e.g., 'D10')
    
    Returns:
        DataFrame with the extracted zone
    """
    upper_left = excel_cell_to_coordinates(upper_left_cell)
    lower_right = excel_cell_to_coordinates(lower_right_cell)
    
    row_start, col_start = upper_left
    row_end, col_end = lower_right
    

    if hasattr(excel_file, 'seek'):
        excel_file.seek(0)
   
    df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
    
    extracted_zone = df.iloc[row_start:row_end+1, col_start:col_end+1].copy()
    if not extracted_zone.empty and not extracted_zone.iloc[0].isnull().all():
        extracted_zone.columns = extracted_zone.iloc[0]
        extracted_zone = extracted_zone.drop(extracted_zone.index[0])
        extracted_zone = extracted_zone.reset_index(drop=True)
    extracted_zone = extracted_zone.dropna(axis=1, how='all')
    
    return extracted_zone

def validate_excel_file(excel_file, required_sheets) -> bool:
    """
    Validate that the Excel file contains the required sheets.
    
    Args:
        excel_file: Excel file path or file-like object
    
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        if hasattr(excel_file, 'seek'): 
            excel_file.seek(0)
        excel_sheets = pd.ExcelFile(excel_file).sheet_names
        
        missing_sheets = [sheet for sheet in required_sheets if sheet not in excel_sheets]
        
        if missing_sheets:
            st.error(f"Feuilles manquantes dans le fichier Excel: {', '.join(missing_sheets)}")
            return False
        
        return True
        
    except Exception as e:
        st.error(f"Erreur lors de la validation du fichier Excel: {str(e)}")
        return False

  
def generate_highlighted_seismic_table_image_data(sismicite_value, type_batiment_value):
    """Generates the highlighted seismic norm table cell image data."""
    batiment_coord_map = {"I": 0, "II avec application possible des PS-MI / CP-MI": 1, "II": 2, "III": 3, "IV": 4}
    sismic_coord_map_for_table = {
        "Zone 1 (très faible)": 0, "Zone 2 (faible)": 1, "Zone 3 (modérée)": 2,
        "Zone 4 (moyenne)": 3, "Zone 5 (forte)": 4
    }
    IMAGE_PATH_NORM = './resources/background.jpg' 

    i_idx_table = sismic_coord_map_for_table.get(sismicite_value, -1)
    j_idx_table = batiment_coord_map.get(type_batiment_value, -1)

    i_coord_for_func = i_idx_table + 1
    j_coord_for_func = j_idx_table + 1
    
    if not os.path.exists(IMAGE_PATH_NORM):
        st.toast(f"Fichier image de fond du tableau sismique introuvable: {IMAGE_PATH_NORM}", icon="🛑")
        return None 

    img_norm_table = cv2.imread(IMAGE_PATH_NORM)
    if img_norm_table is None:
        st.toast(f"Échec du chargement de l'image de fond du tableau sismique: {IMAGE_PATH_NORM}", icon="🛑")
        return None 

    top_left, lower_right = get_coords_to_highlight(i_coord_for_func, j_coord_for_func)
    highlighted_img_data = img_norm_table.copy()
    cv2.rectangle(highlighted_img_data, top_left, lower_right, (0, 0, 255), 3)
    return highlighted_img_data

class DocumentGenerator:
    """Handles document generation logic"""
    
    def __init__(self, template_path):
        self.template_path = template_path
        self.temp_files = [] 
    
    def prepare_images(self, session_state):
        """Prepare all images as temporary files and return their paths."""
        image_paths = {}

        if 'plan_masse' in session_state and session_state.plan_masse is not None:
            temp_plan_orig_path = self._save_plan_masse(session_state.plan_masse)
            if temp_plan_orig_path:
                fixed_plan_path = "./resources/temp_plan_de_masse.png"
                shutil.copy(temp_plan_orig_path, fixed_plan_path)
                image_paths["plan_masse"] = fixed_plan_path

        if hasattr(session_state, 'commune_map_buffer') and session_state.commune_map_buffer:
            path = "./resources/temp_carte_de_sismicite.png"
            with open(path, "wb") as f:
                session_state.commune_map_buffer.seek(0)
                f.write(session_state.commune_map_buffer.read())
            image_paths["carte_sismicite"] = path
        
        if hasattr(session_state, 'highlighted_image') and session_state.highlighted_image is not None:
            path = "./resources/temp_highlighted.png"
            if isinstance(session_state.highlighted_image, np.ndarray ): # Basic check for numpy array from cv2
                 cv2.imwrite(path, session_state.highlighted_image)
                 image_paths["bâtiment_et_sismicité"] = path
            else:
                 st.write(f"{type(session_state.highlighted_image)}\n\n")
                 st.warning("Données pour 'highlighted_image' non valides pour la sauvegarde.")

        beton_type_str = session_state.get("beton_type", "")
        beton_image_path = self._get_image_path("beton", beton_type_str)
        if beton_image_path and os.path.exists(beton_image_path):
            image_paths["image_beton"] = beton_image_path
        
        type_acier_str = session_state.get("type_acier", "")
        acier_image_path = self._get_image_path("acier", type_acier_str)
        if acier_image_path and os.path.exists(acier_image_path):
            image_paths["image_acier"] = acier_image_path
            
        return image_paths
    
    def _save_plan_masse(self, plan_masse_file):
        """Save plan de masse to temporary file, handling both images and PDFs"""
        temp_plan = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        self.temp_files.append(temp_plan.name) 
        
        try:
            plan_masse_file.seek(0) 
            if plan_masse_file.type.startswith('image'):
                temp_plan.write(plan_masse_file.getvalue())
            elif plan_masse_file.type == 'application/pdf':
                converted_image_bytesio = convert_pdf_to_image(plan_masse_file)
                temp_plan.write(converted_image_bytesio.getvalue())
            else:
                st.error(f"Type de fichier non supporté pour plan de masse: {plan_masse_file.type}")
                return None
            
            temp_plan.close()
            return temp_plan.name
        except Exception as e:
            st.error(f"Erreur lors de la sauvegarde du plan de masse: {str(e)}")
            if not temp_plan.closed:
                temp_plan.close()
            return None
    
    def _get_size_multiplier(self, image_type):
        """Get the size multiplier for different image types"""
        size_multipliers = {
            "temp_plan_de_masse": (1.0, 1.5),
            "temp_carte_de_sismicite": (1.0/0.9, 1.1),
            "temp_highlighted": (1.0/0.75, 0.7)
        }
        return size_multipliers.get(image_type, (1.0, 1.0))
    
    def generate_document(self, context):
        """Generate the final document"""
        try:
            doc = DocxTemplate(self.template_path)

            # Apply "hacky fix" to all images with size multipliers
            image_mappings = [
                ("placeholder_image_1", "./resources/temp_plan_de_masse.png", "temp_plan_de_masse"),
                ("placeholder_image_2", self._get_image_path("beton", context.get("beton_type", "")), "beton"),
                ("placeholder_image_3", "./resources/temp_carte_de_sismicite.png", "temp_carte_de_sismicite"),
                ("placeholder_image_4", "./resources/temp_highlighted.png", "temp_highlighted"),
            ]
            
            for placeholder_key, image_path, image_type in image_mappings:
                if image_path and os.path.exists(image_path):
                    try:
                        width_height_proportion, size_multiplier = self._get_size_multiplier(image_type)
                        
                        if size_multiplier != 1.0:
                            # Create InlineImage with custom size
                            inline_img = InlineImage(doc, image_path, width=Mm(100.0 * size_multiplier * width_height_proportion ), height=Mm(100.0 * size_multiplier/ width_height_proportion ))
                        else:
                            # Create InlineImage with default size
                            inline_img = InlineImage(doc, image_path)
                        context[placeholder_key] = inline_img
                    except Exception as e:
                        st.warning(f"Could not process image {image_path}: {str(e)}")
                        continue
            
            doc.render(context)
            
            docx_file = BytesIO()
            doc.save(docx_file)
            docx_file.seek(0)
            
            return docx_file
            
        except FileNotFoundError:
            st.error("Template non trouvé. Vérifiez que 'im_template.docx' existe.")
            return None
        except Exception as e:
            st.error(f"Erreur lors de la génération du document: {str(e)}")
            return None
    
    def _get_image_path(self, material_type, value):
        """Convert material specification to image path and notify if it exists"""
        value_str = str(value)
        
        if material_type == "beton":
            beton_types_simplified = ['C30/37', 'C35/45', 'C40/45', 'C25/30', 'C16/20',]
            for el in beton_types_simplified:
               if el in value_str:
                 path = f"resources/{el.replace('/', '-')}.png"
                 break
            else:
                path = f"resources/{value_str.replace('/', '-')}.png"
        elif material_type == "acier":
            path = f"resources/{value_str}.png"
        else:
            return ""
    
        if not os.path.exists(path):
            st.toast(f"Image not found: {path}")
        return path

    def cleanup(self):
        """Clean up temporary files"""
        for temp_file_path in self.temp_files:
            try:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
            except Exception:
                pass 
        self.temp_files = [] 

        temp_images_to_remove = ["./resources/temp_plan_de_masse.png",
                                 "./resources/temp_carte_de_sismicite.png",
                                 "./resources/temp_highlighted.png"]
        for temp_img_path in temp_images_to_remove:
            try:
                if os.path.exists(temp_img_path):
                    os.unlink(temp_img_path)
            except Exception:
                pass

class FormValidator:
    """Handles form validation logic"""
    
    @staticmethod
    def validate_affaire_number(affaire):
        """Validate that affaire is a 5-digit number"""
        return affaire.isdigit() and len(affaire) == 5
    
    @staticmethod
    def validate_required_fields(session_state):
        """Validate all required fields"""
        errors = []
        
        if not session_state.affaire or not FormValidator.validate_affaire_number(session_state.affaire):
            errors.append("Le numéro d'affaire doit être composé de 5 chiffres exactement.")
        
        if not session_state.commune or session_state.commune.strip() == "":
            errors.append("Veuillez entrer une commune.")
        
        if not session_state.get('calculs_excel'):
            errors.append("Veuillez téléverser le fichier de calcul de portance (.xlsm / .xlsx).")
        
        if not session_state.get('plan_masse'):
            errors.append("Veuillez téléverser le plan de masse.")
            
        return errors

class DataManager:
    """Handles data loading and management with dependency injection"""
    
    def __init__(self, session_state):
        self.session_state = session_state
    
    def load_defaults(self):
        """Load default values from JSON file"""
        defaults = {
            "nom": "", "commune": "", "soustitre": "", 
            "technique": "Forés Tarière Creuse", "affaire": "",
            "maitre": "", "promoteur": "", "geotechnicien": "",
            "beton_type": "C30/37", "type_acier": "Acier HA - Fe E500",
            "diameter_option": "Un seul diamètre", "sismicite": "Zone 1 (très faible)",
            "type_batiment": "I", "documents_marche": [],
            "calculs_excel": None, "plan_masse": None, "horiz_excel": None,
            "highlighted_image": None, "commune_map_buffer": None,
            "selection_synthese": [], "selection_resultats": [],
            "selection_diameters": [], "selected_dataframe": None,
            "selection_horizontal": [], "proceed_to_main": False
        }
        
        try:
            if os.path.exists("./resources/defaults.json"):
                with open("./resources/defaults.json", "r", encoding="utf-8") as f:
                    loaded_defaults = json.load(f)
                    defaults.update(loaded_defaults)
        except Exception as e:
            st.toast(f"Erreur lors du chargement des valeurs par défaut: {str(e)}", icon="⚠️")
        
        for key in ["selection_horizontal","selection_synthese", "selection_resultats", "selection_diameters", "selected_dataframe"]:
            if key not in defaults or defaults[key] is None:
                defaults[key] = []
        
        self.session_state['defaults'] = defaults
        return defaults
    
    def initialize_session_state(self, defaults):
        """Initialize session state with default values"""

        geographic_keys = ['selected_gdf', 'selected_name', 'selected_departement', 
                          'insee_code', 'commune_map_buffer', 'highlighted_image']
        
        for key in geographic_keys:
            if key not in self.session_state:
                self.session_state[key] = defaults.get(key)
        
        session_vars = [
            "nom", "commune", "soustitre", "technique", "affaire", "maitre", 
            "promoteur", "geotechnicien", "beton_type", "type_acier",
            "diameter_option", "sismicite", "type_batiment", "documents_marche",
            "calculs_excel", "plan_masse", "horiz_excel","selection_horizontal", "selection_synthese",
            "selection_resultats", "selection_diameters", "selected_dataframe", "proceed_to_main"
        ]
        
        for var in session_vars:
            if var not in self.session_state:
                self.session_state[var] = defaults[var]
        
        if "documents" not in self.session_state:
            self.session_state.documents = pd.DataFrame({
                "Documents de Marché": self.session_state.documents_marche or [""]
            })
        
        for key in ['display_dataframes', 'generated_image_paths']:
            if key not in self.session_state:
                self.session_state[key] = {}
        
        default_commune = defaults.get("commune")
        if (default_commune and 
            self.session_state.selected_gdf is None and 
            "(" in default_commune):
            self.process_default_commune(default_commune)
    
    def process_default_commune(self, commune_string):
        """Process commune from defaults to populate geographic data"""
        try:
            if "(" in commune_string and ")" in commune_string:
                insee_code = commune_string.split("(")[-1].split(")")[0]
                commune_name = commune_string.split(" (")[0]
                
                shapefile_path = "./resources/france_zonage_sismique/France_zonage_sismique.shp"
                data = gpd.read_file(shapefile_path) 
                
                if data is not None:
                    match = data[data['insee'] == insee_code]
                    if not match.empty:
                        self.session_state.insee_code = insee_code
                        self.session_state.selected_gdf = match.iloc[[0]]
                        self.session_state.selected_name = commune_name
                        self.session_state.selected_departement = str(match.iloc[0]['departemen'])
                        
                        seismic_zones = {
                            '1 - Très faible': "Zone 1 (très faible)",
                            '2 - Faible': "Zone 2 (faible)",
                            '3 - Modérée': "Zone 3 (modérée)",
                            '4 - Moyenne': "Zone 4 (moyenne)",
                            '5 - Forte': "Zone 5 (forte)"
                        }
                        if 'Sismicite' in match.columns:
                            sismicite_value = match.iloc[0]['Sismicite']
                            if sismicite_value in seismic_zones:
                                self.session_state.sismicite = seismic_zones[sismicite_value]
                                
        except Exception as e:
            st.warning(f"Impossible de traiter la commune par défaut '{commune_string}': {e}")
    
    def get_uploaded_files_summary(self):
        """Get summary of uploaded files"""
        summary = {}
        
        calculs_excel = self.session_state.get('calculs_excel')
        plan_masse = self.session_state.get('plan_masse')
        horiz_excel = self.session_state.get('horiz_excel')
        
        summary['calculs_excel'] = calculs_excel.name if calculs_excel else None
        summary['plan_masse'] = plan_masse.name if plan_masse else None
        summary['horiz_excel'] = horiz_excel.name if horiz_excel else None
        summary['has_required_files'] = bool(calculs_excel and plan_masse)
        
        return summary

def horizontal_details_un_pieux(df: pd.DataFrame) -> dict:

    def get_range(cell_start: str, cell_end: str) -> list[tuple[int, int]]:
        r1, c1 = excel_cell_to_coordinates(cell_start)
        r2, c2 = excel_cell_to_coordinates(cell_end)
        return [(r, c1) for r in range(r1, r2 + 1)]

    horiz_list = [
        'num', 'vmin', 'diam', 'arase','longueur', 'elast',
        'pas_calcul', 'pas_sortie', 'inertie', 'type',
        'tete', 'T', 'M', 'delta', 'Max', 'Tmax',
    ]

    ranges = [
        get_range("K3", "K11"),
        get_range("P3", "P6"),
        get_range("V3", "V5"),
    ]

    all_coords = list(chain.from_iterable(ranges))

    if len(horiz_list) != len(all_coords):
        raise ValueError("Mismatch between horiz_list and cell range size.")

    values = [df.iat[r, c] for r, c in all_coords]
    return dict(zip(horiz_list, values))


def format_if_numeric(x):
    if pd.isna(x):
        return ""

    if isinstance(x, str):
        x = x.replace(',', '.')

    try:
        return f"{float(x):.2f}"
    except (ValueError, TypeError):
        return str(x)

def init_select_column(df: pd.DataFrame) -> pd.DataFrame:
    """Initialize the dataframe with a selection column and a hash column"""
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", False)

    df_with_selections["hash"] = df_with_selections.apply(
        lambda x: md5("|".join(tuple(str(x))).encode()).hexdigest(), axis=1
    )

    # Initialize the selection status in the session state 
    # with the hash as the key and the selection status as the value
    if "select_status" not in st.session_state:
        st.session_state.select_status = df_with_selections[["Select", "hash"]].set_index("hash")["Select"].to_dict()
    
    return df_with_selections

def dataframe_with_selections(df: pd.DataFrame) -> pd.DataFrame:
    """Display the dataframe with a selection column and allow the user to select rows"""

    st.write("📋 Selectionnez les lignes à inclure dans le ndc:")
    
    edited_df = st.data_editor(
        df,
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        use_container_width=True,
        num_rows="fixed",
    )

    if "select_status" in st.session_state:
        for _, (select, hash_val) in edited_df[['Select', "hash"]].iterrows():
            st.session_state["select_status"][hash_val] = select

    selected_rows = edited_df[edited_df.Select]
    return selected_rows.drop(['Select', 'hash'], axis=1)

def rename_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames duplicate columns in a DataFrame by appending an index.
    e.g., 'col' becomes 'col_0', 'col' becomes 'col_1'
    """
    cols = df.columns.tolist()
    seen = {}
    new_cols = []
    for col in cols:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}.{seen[col]}")
        else:
            seen[col] = 0 # Initialize with 0 to make first duplicate 'col_1'
            new_cols.append(col)
    df.columns = new_cols
    return df

def move_nones_to_end(df: pd.DataFrame) -> pd.DataFrame:
    non_nan_cols = df.columns[df.notna().any()].tolist()
    nan_cols = [col for col in df.columns if col not in non_nan_cols]
    return df[non_nan_cols + nan_cols]

def restore_column_order(original_df: pd.DataFrame, edited_df: pd.DataFrame) -> pd.DataFrame:
    # Update only columns that were present in edited_df
    for col in edited_df.columns:
        original_df[col] = edited_df[col]
    return original_df

def instantiate_dataframes(session_state):
    try:
        if hasattr(session_state.calculs_excel, 'seek'):
            session_state.calculs_excel.seek(0)
        processed_tables_for_doc_context = {}

        # Géotechnique            
        df_hyp_raw = extract_excel_zone(session_state.calculs_excel, 'Hypothèses', 'D6', 'P18')
        display_df_hyp = pd.DataFrame()
        if not df_hyp_raw.empty:
            if 'Type Sol\n(NF P 94-262)' in df_hyp_raw.columns:
                last_valid_idx = df_hyp_raw['Type Sol\n(NF P 94-262)'].last_valid_index()
                if last_valid_idx is not None:
                    df_hyp_filtered_for_doc = df_hyp_raw.loc[:last_valid_idx].copy()
                    df_hyp_filtered_for_doc.rename(columns={'Type Sol\n(NF P 94-262)': 'Type Sol'}, inplace=True)

                    aliases = {'kp / kc max': 'kp max','pl*/qc (MPa)': 'pl* (MPa)'} 
                    if any(item for item in aliases.keys() if item in df_hyp_raw.columns):
                        # a little repetition rather than a little abstraction
                        df_hyp_filtered_for_doc.rename(columns= aliases, inplace=True)
                    
                    display_df_hyp = df_hyp_filtered_for_doc.applymap(format_if_numeric).copy()
                else:
                    st.toast("La colonne 'Type Sol\\n(NF P 94-262)' est vide dans 'Hypothèses'.", icon="⚠️")
            else:
                display_df_hyp = df_hyp_raw.copy()
                st.toast("Colonne 'Type Sol\\n(NF P 94-262)' non trouvée dans 'Hypothèses'.", icon="⚠️")
        else:
            st.toast("Impossible d'extraire les données géotechniques de la feuille 'Hypothèses' (plage D6:P18).", icon="⚠️")
        processed_tables_for_doc_context["geodata"] = display_df_hyp

        # Portance data & Synthèse
        verif_df_raw  = pd.read_excel(session_state.calculs_excel, sheet_name="Dimensionnement Général", skiprows=11, engine='openpyxl')
        verif_df_raw = verif_df_raw.dropna( axis=0, how='all')
        #verif_df_raw = extract_excel_zone(session_state.calculs_excel, 'Dimensionnement Général', 'A12', 'AX126')
        display_df_verif = pd.DataFrame()
        if not verif_df_raw.empty:
            display_df_verif = verif_df_raw.applymap(format_if_numeric).copy()
        else:
            st.toast("Aucune donnée de vérification de portance extraite de 'Dimensionnement Général'.", icon="⚠️")

        if session_state.horiz_excel:
            horizontal = pd.read_excel(session_state.horiz_excel, sheet_name="Cas Multiples", skiprows=6, engine='openpyxl')
            display_df_verif= pd.concat([display_df_verif, rename_duplicate_columns(horizontal)], axis=1)

        processed_tables_for_doc_context["verifications"] = display_df_verif.apply(
                                                                                   lambda col: col.where( col.index <= col.last_valid_index()),
                                                                                   axis=0
                                                                               ).round(2)

        # Optional
        try:
            diam_df_raw = pd.read_excel(session_state.calculs_excel, sheet_name="Béton", skiprows=9, engine='openpyxl').ffill()
            diam_df_raw =diam_df_raw.dropna( axis=1, how='all')

            display_df_diam = pd.DataFrame()
            if not diam_df_raw.empty:
                display_df_diam = diam_df_raw.applymap(format_if_numeric).copy()
            else:
                st.toast("Aucune donnée de diamètre extraite de 'Béton'.", icon="⚠️")
            processed_tables_for_doc_context["diameters"] = rename_duplicate_columns( display_df_diam )
        except:
            pass #processed_tables_for_doc_context["diameters"] = pd.DataFrame(),

        return processed_tables_for_doc_context

    except Exception as e:
        st.error(f"Erreur majeure lors de l'extraction des données Excel: {str(e)}")
        return {
            "geodata": pd.DataFrame(),
            "verifications": pd.DataFrame(),
            "diameters": pd.DataFrame(),
        }


def build_document_context(session_state):
    """Build context dictionary for document generation and populate display_dataframes."""
    technique_details = TECHNIQUES.get(session_state.technique, {
        "description_technique": "technique non spécifiée",
        "norme_associe": "norme applicable non déterminée"
    })
    
    beton_details = BETON_TYPES.get(session_state.beton_type, "spécification béton non déterminée")
    if session_state.diameter_option == "Plusieurs diamètres":
        beton_details += " Adaptation pour plusieurs diamètres."
    
    acier_details = ACIER_TYPES.get(session_state.type_acier, "Spécification acier non déterminée")
    
    technique_full = f"Pieux {session_state.technique}"
    introduction = f"""pieux seront réalisés suivant la technique des {technique_details['description_technique']}. 
Au sens de la {technique_details['norme_associe']}"""

    essais = """
    Pour ce chantier, il sera réalisé :
    \t- Série d’éprouvette béton.
    \t- Enregistrement des paramètres de forage.
    """
    
    # Count the number of 'Oui' entries in the 'CR Utile' column (case-insensitive)
    if 'CR Utile' in session_state.display_dataframes['verifications'].columns:
        count_oui = (session_state.display_dataframes['verifications']['CR Utile']
                     .str.lower() == 'oui').sum()
    
        if count_oui > 0:
            essais += (
                f"\n- {count_oui} pieux sont concernés par un contrôle renforcé ; "
                f"{math.ceil(count_oui / 4)} essais d'impédance seront réalisés parmi les pieux ci-dessus."
            )
    
    context = {
        "NOM": session_state.nom, 
        "COMMUNE": session_state.commune,
        "soustitre": session_state.soustitre,
        "technique": technique_full,
        "introduction": introduction,
        "essais": essais,
        "ref_docs": session_state.documents_marche,
        "beton": beton_details,
        "acier": acier_details,
        "affaire": session_state.affaire,
        "maitre": session_state.maitre,
        "promoteur": session_state.promoteur,
        "geotechnicien": session_state.geotechnicien,
        "sismicite": session_state.sismicite,
        "type_batiment": session_state.type_batiment,
        "beton_type": session_state.beton_type, 
        "type_acier": session_state.type_acier,
        "selection_synthese": session_state.selection_synthese,
        "selection_resultats": session_state.selection_resultats,
        "selection_diameters": session_state.selection_diameters,
        "selected_dataframe": session_state.selected_dataframe,
    }

    if session_state.diameter_option == "Plusieurs diamètres":
        session_state.display_dataframes['diameters'] = session_state.selected_dataframe if not session_state.selected_dataframe.empty else session_state.display_dataframes['diameters'].head()

    if session_state.horiz_excel:
        #singleton I guess.
        context["horiz"] = horizontal_details_un_pieux( pd.read_excel(session_state.horiz_excel, sheet_name='Calcul Horiz'))
        context["selection_horizontal"] = session_state.selection_horizontal

    #I'm simply assuming if the program reached here the rendering of Détails des pieux worked sans soucis
    for el in session_state.display_dataframes:
        context[el] = session_state.display_dataframes[el].apply(lambda col: col.where( col.index <= col.last_valid_index()),axis=0).to_dict( 'records' )

    return context
