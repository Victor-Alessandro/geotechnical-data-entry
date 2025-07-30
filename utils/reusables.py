# This is meant to functions used in more than one page. ndc, ddc, sol... are meant to the main page and their respective ones as well.

import streamlit as st
import pandas as pd
import subprocess
import tempfile
import shutil
import uuid


from pathlib import Path
from typing import TypedDict, Optional, List, Dict, Any

# I've been having trouble with Streamlits choice of scope for st.session_state
# I would much rather have immutability be the default, that leads to most variables being consistent
# but that isn't the default. Coupled with the fact that session_state is global within the scope of a page, but can also be shared
# I felt the needed of creating myself some guardrails so their scope is always clear. This means centralization.
AppSessionState = {
    "DDC": TypedDict("DDC", {
        "ddc_uploaded": bool,
        "ddc_tables": Optional[Any],
        "selected_ddc_table": Optional[Any],
        "formatted_ddc_table": Optional[Any],
        "crop_params": Dict[str, int],
        "final_ddc_data": Optional[Any],
    }),
    "Geo": TypedDict("Geo", {
        "nom": str,
        "commune": str,
        "soustitre": str,
        "technique": str,
        "affaire": str,
        "maitre": str,
        "promoteur": str,
        "geotechnicien": str,
        "beton_type": str,
        "type_acier": str,
        "diameter_option": str,
        "sismicite": str,
        "type_batiment": str,
        "documents_marche": List[str],
        "calculs_excel": Optional[Any],
        "plan_masse": Optional[Any],
        "highlighted_image": Optional[Any],
        "commune_map_buffer": Optional[Any],
        "selected_gdf": Optional[Any],
        "selected_name": Optional[str],
        "selected_departement": Optional[str],
        "insee_code": Optional[str],
        "documents": pd.DataFrame,
        "display_dataframes": List[Any],
        "generated_image_paths": Dict[str, str],
    }),
    "extraction": TypedDict("extraction", {
        "files_uploaded": bool,
        "etude_sol_uploaded": bool,
        "document_tables": List[Any],
        "table_ratings": Dict[str, float],
        "selected_tables": pd.DataFrame,
    }),
    "generation": TypedDict("generation", {}),
    "shared": TypedDict("Shared", {
        "page_info": Dict[str, Any],
        "all_extracted_tables": Dict[str, Any],
        "max_workers": int,
        "formatted_table": Optional[pd.DataFrame],
        "final_dataframe": Optional[pd.DataFrame],
        "defaults": Dict[str, Any],
    }),
}



def restart_page():
  #the goal is to initialize state on the main app ( Téléchargements ) and have each page be as read only as possible.
  for key in list(st.session_state.keys()):
      del st.session_state[key]
  st.rerun()

def initialize_session_state( state_dictionary ):
    """Initialize session state variables shared across pages"""
    for key, value in state_dictionary.items():
        if key not in st.session_state:
            st.session_state[key] = value



# If I do this the memory footprint is reduced, but for large files the processing is slow
def strip_pdf(input_file) -> bytes:
    """
    Take a PDF file-like object, flatten all interactive features,
    convert to grayscale, and return stripped PDF bytes.
    Requires Ghostscript installed on the system.
    """

    # if ghostscript is not available return the bytes from the original pdf
    if shutil.which( "gs" ) is None:
        return input_file.getvalue()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_in:
        tmp_in.write(input_file.read())
        tmp_in.flush()
        in_path = tmp_in.name


    unique_name = f"stripped_{uuid.uuid4().hex}.pdf"
    out_path = Path(tempfile.gettempdir()) / unique_name

    gs_cmd = [
        "gs",
        "-sDEVICE=pdfwrite",                  
        "-dCompatibilityLevel=1.4",           
        "-dPDFSETTINGS=/screen",              
        "-dProcessColorModel=/DeviceGray",    
        "-dColorConversionStrategy=/Gray",
        "-dColorConversionStrategyForImages=/Gray",
        "-dNOPAUSE",                          
        "-dBATCH",                            
        "-dQUIET",                            
        f"-sOutputFile={out_path}",
        in_path
    ]

    try:
        subprocess.run(gs_cmd, check=True)
    except subprocess.CalledProcessError as e:
        st.error(f"Error stripping PDF: {e}")
        return None

    with open(out_path, "rb") as f:
        stripped_bytes = f.read()

    try:
        Path(in_path).unlink()
        out_path.unlink()
    except Exception:
        pass

    return stripped_bytes
