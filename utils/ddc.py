import streamlit as st
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
import threading
import camelot
import gc

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Tuple, List
from functools import lru_cache

class ProgressTracker:
    def __init__(self, total_pages: int):
        self.total_pages = total_pages
        self.completed_pages = 0
        self.lock = threading.Lock()
        
    def update(self):
        with self.lock:
            self.completed_pages += 1
            return self.completed_pages, self.total_pages
    
    def get_progress(self):
        with self.lock:
            return self.completed_pages, self.total_pages


@lru_cache(maxsize=128)
def detect_paper_size(width_mm: float, height_mm: float) -> str:
    """
    Détecte le format du papier en fonction des dimensions en mm (avec cache LRU).
    
    Args:
        width_mm: Largeur en millimètres
        height_mm: Hauteur en millimètres
        
    Returns:
        Chaîne indiquant le format du papier
    """
    paper_sizes = {
        "A0": (841, 1189),
        "A1": (594, 841),
        "A2": (420, 594),
        "A3": (297, 420),
        "A4": (210, 297),
        "A5": (148, 210),
        "Lettre": (216, 279),
        "Légal": (216, 356),
        "Tabloïd": (279, 432)
    }
    
    for name, (std_width, std_height) in paper_sizes.items():
        if (abs(width_mm - std_width) < std_width * 0.05 and 
            abs(height_mm - std_height) < std_height * 0.05):
            return f"{name} Portrait"
        
        if (abs(width_mm - std_height) < std_height * 0.05 and 
            abs(height_mm - std_width) < std_width * 0.05):
            return f"{name} Paysage"
    
    return f"Personnalisé ({width_mm:.1f} x {height_mm:.1f} mm)"

def convert_pdf_to_image(pdf_path: str, page_num: int = 0, dpi: int = 120) -> Tuple[bytes, str, Optional[np.ndarray]]:
    """
    Convertir une page PDF en image en mémoire, avec option de prétraitement optimisé.
    
    Args:
        pdf_path: Chemin vers le fichier PDF
        page_num: Numéro de page à convertir
        dpi: DPI pour la conversion d'image
        
    Returns:
        Tuple (octets de l'image, format du papier, image prétraitée comme tableau numpy si applicable)
    """
    doc = None
    pixmap = None
    
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        matrix = fitz.Matrix(dpi/72, dpi/72)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        
        img_bytes = pixmap.tobytes("png")
        
        width, height = page.rect.width, page.rect.height
        width_mm = width * 0.3528
        height_mm = height * 0.3528
        paper_size = detect_paper_size(width_mm, height_mm)
        
        return img_bytes, paper_size
        
    except Exception as e:
        raise Exception(f"Erreur lors de la conversion PDF->Image: {e}")
    finally:
        if pixmap:
            pixmap = None
        if doc:
            doc.close()
        gc.collect()


def clean_large_table(df: pd.DataFrame, keep_threshold: float = 0.05) -> pd.DataFrame:
    def is_numeric(val):
        try:
            float(val)
            return True
        except:
            return False

    def clean_cell(cell):
        if isinstance(cell, str) and not is_numeric(cell) and len(cell) > 12:
            return float( 'NaN')

        #if cell == "":
        #    return float( 'NaN')
        if isinstance(cell, str) and not cell.strip():
            return float( 'NaN' )
              
        return cell

    cleaned = df.applymap(clean_cell)

    cleaned = cleaned.dropna(axis=0, thresh=int(keep_threshold * cleaned.shape[1]))
    cleaned = cleaned.dropna(axis=1, thresh=int(keep_threshold * cleaned.shape[0]))

    cleaned = cleaned.dropna( axis=0, how="all")
    cleaned = cleaned.dropna( axis=1, how="all")

    return cleaned

def process_single_page(args: Tuple[str, int, str, str, int]) -> Tuple[int, List[Dict], str]:
    """
    Traiter une seule page pour l'extraction de tableaux (pour le traitement parallèle).
    
    Args:
        args: Tuple contenant (pdf_path, page_num, flavor, global_table_index_start)
        
    Returns:
        Tuple (page_num, liste des informations de tableau, format du papier)
    """
    pdf_path, page_num, flavor, global_table_index_start = args
    
    try:
        _, paper_size = convert_pdf_to_image(
            pdf_path, page_num, dpi=120
        )
        
        extraction_source = pdf_path

        tables = camelot.read_pdf(
                extraction_source,
                pages=str(page_num + 1),
                flavor=flavor
            )
        
        page_tables = []
        for i, table in enumerate(tables):
            table_info = {
                'dataframe': table.df,
                'global_index': global_table_index_start + i,
                'page_index': i,
                'page_number': page_num + 1,
                'paper_size': paper_size,
                'table_title': f"Tableau {global_table_index_start + i + 1} (Page {page_num + 1}, Tableau {i + 1})"
            }
            page_tables.append(table_info)
       
        return page_num, page_tables, paper_size
        
    except Exception as e:
        st.warning(f"Erreur lors du traitement de la page {page_num + 1}: {e}")
        return page_num, [], "Erreur"

def extract_tables_from_pdf_parallel(pdf_path: str, page_nums: List[int] = None, flavor: str = "stream", 
                                    max_workers: int = 2) -> Tuple[Dict, List, Dict]:
    """
    Extraire les tableaux du PDF en utilisant le traitement parallèle.
    
    Args:
        pdf_path: Chemin vers le fichier PDF
        page_nums: Liste des numéros de page à extraire
        flavor: Méthode d'extraction ('lattice' ou 'stream')
        max_workers: Nombre de workers pour le traitement parallèle
        
    Returns:
        Tuple (Dictionnaire des tableaux par page, Liste des tables camelot, Info des pages)
    """
    # Conversion du flavor
    if flavor == "Séparée par Lignes":
        flavor = "lattice"
    else:
        flavor = "stream"
    
    if page_nums is None:
        page_nums = [0]
    
    progress_tracker = ProgressTracker(len(page_nums))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        all_tables = {}
        all_camelot_tables = []
        page_info = {}
        global_table_index = 0
        
        tasks = []
        for page_num in page_nums:
            args = (pdf_path, page_num, flavor, global_table_index)
            tasks.append(args)
            global_table_index += 3  
        
        global_table_index = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_page = {executor.submit(process_single_page, task): task[1] for task in tasks}
            
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    page_num_result, page_tables, paper_size = future.result()
                    
                    for table_info in page_tables:
                        table_info['global_index'] = global_table_index
                        table_info['table_title'] = f"Tableau {global_table_index + 1} (Page {page_num_result + 1}, Tableau {table_info['page_index'] + 1})"
                        global_table_index += 1
                    
                    all_tables[page_num_result] = page_tables
                    page_info[page_num_result] = paper_size
                    
                    for table_info in page_tables:
                        all_camelot_tables.append(table_info)                   

                    completed, total = progress_tracker.update()
                    progress = completed / total
                    progress_bar.progress(progress)
                    status_text.text(f"Traitement: {completed}/{total} pages terminées")
                    
                except Exception as e:
                    st.error(f"Erreur lors du traitement de la page {page_num + 1}: {e}")
        
        progress_bar.empty()
        status_text.empty()
        
        return all_tables, all_camelot_tables, page_info
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Erreur lors de l'extraction parallèle des tableaux: {e}")
        return {}, [], {}
