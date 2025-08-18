import json
import os
import io
import streamlit as st
import pandas as pd

from streamlit_extras.pdf_viewer import pdf_viewer
from typing import List, Dict, Any
from PyPDF2 import PdfReader

from utils import geo4 as geo

# ==============================================================================
# 1. UI COMPONENTS
# ==============================================================================

def display_table_analysis_results(results: List[Dict[str, Any]], num_to_show: int) -> List[str]:
    """Display analysis results and return list of selected tables."""
    top_results = results[:num_to_show]
    
    if not top_results:
        st.warning("Aucun tableau ne correspond aux critères de recherche après filtrage.")
        return []
    
    selected_tables = []
    for result_rank, result in enumerate(top_results):
        df = result['df']
        page = result['page']
        
        with st.container(border=True):
            st.markdown(f"**Page {page}**")
            st.data_editor(df, use_container_width=True, key=f"editor_{result['df_index']}")

            rank_key = f"rating_{result['df_index']}"
            default_bool = 0 if result_rank < 3 else 1
            
            selected = st.columns(3)[1].radio(
                'Informations Pertinentes', 
                ['Vrai', 'Faux'], 
                index=default_bool, 
                key=rank_key, 
                horizontal=True
            )

            selected_tables.append(selected)
    
    return selected_tables

@st.fragment
def show_all_tables_fragment():
    """Fragment to show all extracted tables with selection capability."""
    column_indices_map = {}
    for idx, (df, _) in enumerate(st.session_state.document_tables):
        cols_tuple = tuple(df.columns)
        column_indices_map.setdefault(cols_tuple, []).append(idx)
    
    keys_to_dfs = list(column_indices_map.keys())
    if not keys_to_dfs:
        st.warning("Aucune structure de tableau à afficher.")
        return

    max_len = max(len(key) for key in keys_to_dfs)
    normalized_list = [list(inner_list) + [''] * (max_len - len(inner_list)) for inner_list in keys_to_dfs]
    structures_df = pd.DataFrame(normalized_list)

    selection = st.dataframe(
        structures_df, 
        use_container_width=True, 
        hide_index=True,
        on_select='rerun', 
        selection_mode='multi-row'
    )

    if selection and 'rows' in selection.get('selection', {}):
        selected_rows = selection['selection']['rows']
        if st.button("🔍 Afficher les tableaux sélectionnés"):
            if selected_rows:
                selected_indices = []
                for row_idx in selected_rows:
                    clean_key = tuple(filter(lambda a: a != "", normalized_list[row_idx]))
                    if clean_key in column_indices_map:
                        selected_indices.extend(column_indices_map[clean_key])

                for df_idx in sorted(set(selected_indices)):
                    st.write(f"**Tableau (Index d'origine {df_idx}):**")
                    df, _ = st.session_state.document_tables[df_idx]
                    st.dataframe(df, use_container_width=True)

def display_main_interface():
    """Main UI for displaying results and interactions after file upload."""
    keyword_sets = geo.get_keyword_sets()
    
    with st.spinner("Analyse des tableaux en cours..."):
        results = geo.analyze_and_score_tables(st.session_state.document_tables, keyword_sets)

    st.divider()
    st.write("### Tableaux du Document")

    num_to_show = len(results)
    if num_to_show> 1:
        num_to_show = st.slider(
            "Tableaux à afficher :", 
            min_value=1, 
            max_value= num_to_show, 
            value=min(5, num_to_show or 1), 
            step=1
        )

    selected_tables = display_table_analysis_results(results, num_to_show)
    
    if selected_tables and results:
        pertinent_dfs = [
            result['df'] for result, selected in zip(results[:num_to_show], selected_tables) 
            if selected == 'Vrai'
        ]

        df = pd.concat(pertinent_dfs, axis=1) if pertinent_dfs else pd.DataFrame()
        df.columns = geo.truncate_colnames( df.columns ) 
        st.session_state.pertinent_table = df
    
    with st.expander("📂 Afficher tous les tableaux extraits"):
        show_all_tables_fragment()

def handle_json_upload_shortcut():
    """Handle direct JSON file upload as a shortcut."""
    st.header("Charger le document JSON")
    document = st.file_uploader(
        "Choisir le fichier d'étude de sol (format JSON)",
        type=['json'],
        key="etude_json"
    )

    if document and not st.session_state.get("raccourci_loaded"):
        try:
            json_data = json.load(document)
            tables = geo.parse_tables_from_json(json_data)
            if tables:
                st.session_state.document_tables = tables
                st.session_state.etude_sol_uploaded = True
                st.session_state.raccourci_loaded = True
                st.session_state['ocr_result'] = json_data
                st.success(f"🎉 {len(tables)} tableaux extraits du document!")
                st.rerun()
            else:
                st.warning("Aucun tableau trouvé dans le fichier JSON.")
        except json.JSONDecodeError:
            st.error("Erreur: Le fichier n'est pas un JSON valide.")
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier JSON: {str(e)}")


def display_categorized_results(categorized_phrases: Dict[str, List[Dict[str, Any]]]):
    """
    Display the categorized phrases in a 3-column layout, with each match
    as a separate item.
    """
    st.header("Informations Extraites")
    
    active_categories = {
        name: phrases for name, phrases in categorized_phrases.items() if phrases
    }

    if not active_categories:
        st.info("Aucune information pertinente n'a été trouvée selon les règles définies.")
        return

    category_names = list(active_categories.keys())

    cols = st.columns(3)

    for i, name in enumerate(category_names):
        with cols[i % 3]:
            results = active_categories[name]
            
            with st.expander(f"##### {name}"):

                # Loop through each result and display it on a new line
                # with a bullet point for clarity.
                for result in results:
                    st.markdown(
                        f"&bull; {result['highlighted']} *:small[(p. {result['page']})]*", 
                        unsafe_allow_html=True
                    )

def text_extraction_interface():
    """
    Main interface orchestrator.
    """
    if 'nlp' not in st.session_state:
        st.session_state['nlp'] = geo.get_spacy_nlp()
    
    if 'ocr_result' in st.session_state and st.session_state['nlp']:
        nlp = st.session_state['nlp']
        
        if 'extracted_items' not in st.session_state:
            with st.spinner("Analyse du document..."):
                st.session_state['extracted_items'] = geo.extract_items_from_elements(
                    st.session_state['ocr_result'], nlp
                )
        
        items = st.session_state['extracted_items']
        
        categorized_results = geo.run_extraction_pipeline(items, geo.EXTRACTION_CONFIGS, nlp)
        
        display_categorized_results(categorized_results)

    st.divider()
            
    json_filename = "etude_de_sol.json"
    json_str = json.dumps(st.session_state.get('ocr_result', {}), indent=2, ensure_ascii=False)
    st.download_button(
        label="📥 Télécharger le résultat JSON",
        data=json_str,
        file_name=json_filename,
        mime="application/json"
    )


def process_selected_pages( parsing_service, temp_pdf_path, use_ocr ):
    try:
        result, error_msg = parsing_service( temp_pdf_path, use_ocr )
        
        if error_msg:
            st.error(f"❌ Échec du traitement: {error_msg}")
            st.warning("""
            **Suggestions pour résoudre le problème:**
            - Vérifiez que le fichier PDF n'est pas corrompu
            - Essayez de réduire le nombre de pages sélectionnées
            - Si le document contient beaucoup d'images, essayez de désactiver l'OCR temporairement
            - Vérifiez votre connexion internet
            """)

        # result = json data from aryn / chunkr
        elif result:
            st.session_state['ocr_result'] = result
            
            st.success("✅ Document traité avec succès!")
            
            # Parse and store tables
            tables = geo.parse_tables_from_json(result)
            if tables:
                st.session_state.document_tables = tables
                st.session_state.etude_sol_uploaded = True
                st.success(f"🎉 {len(tables)} tableaux extraits du document!")
                st.rerun()
            else:
                st.info("ℹ️ Aucun tableau détecté dans les pages sélectionnées.")
        else:
            st.error("❌ Aucun résultat retourné par l'API")
            
    finally:
        try:
            os.unlink(temp_pdf_path)
        except:
            pass

def handle_pdf_upload_and_analysis():
    """Handle PDF upload, preview, and analysis workflow."""
                   
    st.header("📁 Étape 1 : Charger le document PDF")
    uploaded_pdf = st.file_uploader(
        "Choisir le fichier d'étude de sol (format PDF)",
        type=['pdf'],
        key="etude_sol"
    )
    
    if uploaded_pdf:
        pdf_bytes = uploaded_pdf.getvalue()
            
        reader = PdfReader( io.BytesIO( pdf_bytes) )
        num_pages = len(reader.pages)        

        col1, col2 = st.columns([3, 1])
        
        with col1:
            pdf_viewer(pdf_bytes)
        
        with col2:
            st.info("💡 **Conseil**: Les annexes ne devraient généralement pas être incluses dans l'analyse.")
            
            page_range = st.text_input(
                "Entrez les pages pour l'extraction ", 
                f"0-{num_pages-1}",
                key="page_range",
                help="Exemples: '0,2,4' ou '1-5' ou '0,3-7,10'"
            )
            
            selected_pages = geo.parse_page_range(page_range, num_pages)
            
            use_ocr = st.checkbox(
                "Utiliser l'OCR", 
                value=True, 
                help="Necéssaire pour extraire le texte et tableaux des documents scannés"
            )
            
            if st.button("🔍 Analyser le document avec Aryn", type="primary"):
                with st.spinner("Traitement du document en cours..."):
                    temp_pdf_path = geo.create_filtered_pdf(pdf_bytes, selected_pages)

                    process_selected_pages( geo.process_with_aryn, temp_pdf_path, use_ocr )


            if st.button( "Analyser le document avec Chunkr" ):
                with st.spinner("Traitement du document en cours..."):
                    temp_pdf_path = geo.create_filtered_pdf(pdf_bytes, selected_pages)

                    process_selected_pages( geo.process_with_chunkr, temp_pdf_path, use_ocr )                    

    with st.expander( 'raccourci'):
        handle_json_upload_shortcut()


def table_extraction_interface():
    """Table extraction interface after successful upload."""

    button_container = st.container()
    processing_container = st.container()
    interface_container = st.container()
    
    with button_container:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Changer de fichier"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        with col2:
            extract_button = st.button("Extraire un Modèle Geotéchnique", icon='📈')
            
    st.markdown("---")


    os.environ['GEMINI_API_KEY'] = st.secrets['GOOGLE_API_KEY']
    with processing_container:
        if extract_button:
            simpler_columns = geo.clean_column_names( st.session_state.pertinent_table )         
            result  = geo.placeholder_process_geotechnical_data( simpler_columns )

            st.session_state.final_geo_data = simpler_columns[result.output.names_of_columns_used_for_extraction]

            cols_to_convert = [0, 2]
            for col in cols_to_convert:
                if col < len(st.session_state.final_geo_data.columns):
                    st.session_state.final_geo_data.iloc[:, col] = pd.to_numeric(
                        st.session_state.final_geo_data.iloc[:, col], 
                        errors='ignore'
                    )

            st.dataframe( st.session_state.final_geo_data )
            
    with interface_container:
        display_main_interface()


# ==============================================================================
# 2. MAIN APPLICATION
# ==============================================================================

def landing_page():
    """Application entry point."""
    st.set_page_config(
        page_title="Analyseur de Tableaux de Documents", 
        page_icon="📊", 
        layout='wide'
    )
    
    st.title("📊 Analyseur de Tableaux de Documents")

    geo.initialize_session_state()
    
    if not st.session_state.etude_sol_uploaded:
        handle_pdf_upload_and_analysis()
    else:
        tab1, tab2 = st.tabs(["Analyse des Tableaux", "Information Textuelle"])
        
        with tab1:
            table_extraction_interface()

        with tab2:
            text_extraction_interface()

if __name__ == "__main__":
    landing_page()
