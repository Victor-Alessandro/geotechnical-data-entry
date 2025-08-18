import json
import os
import streamlit as st
import pandas as pd
import fitz

from streamlit_extras.pdf_viewer import pdf_viewer
from typing import List, Dict, Any

from utils import geo

# ==============================================================================
# 2. UI COMPONENTS
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
            st.data_editor(df, use_container_width=True)
            
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

    num_to_show = st.slider(
        "Tableaux à afficher :", 
        min_value=1, 
        max_value=max(1, len(results)), 
        value=min(5, len(results) or 1), 
        step=1
    )

    selected_tables = display_table_analysis_results(results, num_to_show)
    
    # Update pertinent table based on selections
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
                st.session_state['aryn_result'] = json_data
                st.success(f"🎉 {len(tables)} tableaux extraits du document!")
                st.rerun()
            else:
                st.warning("Aucun tableau trouvé dans le fichier JSON.")
        except json.JSONDecodeError:
            st.error("Erreur: Le fichier n'est pas un JSON valide.")
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier JSON: {str(e)}")

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
        
        with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
            num_pages = len(pdf)
        
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
                    
                    try:
                        result, error_msg = geo.process_with_aryn(temp_pdf_path, use_ocr)
                        
                        if error_msg:
                            st.error(f"❌ Échec du traitement Aryn: {error_msg}")
                            st.warning("""
                            **Suggestions pour résoudre le problème:**
                            - Vérifiez que le fichier PDF n'est pas corrompu
                            - Essayez de réduire le nombre de pages sélectionnées
                            - Si le document contient beaucoup d'images, essayez de désactiver l'OCR temporairement
                            - Vérifiez votre connexion internet
                            """)

                        # result = json data from aryn
                        elif result:
                            st.session_state['aryn_result'] = result
                            
                            st.success(f"✅ Document traité avec succès! {len(selected_pages)} pages analysées.")
                            
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
                            st.error("❌ Aucun résultat retourné par l'API Aryn")
                            
                    finally:
                        try:
                            os.unlink(temp_pdf_path)
                        except:
                            pass

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

            st.dataframe( st.session_state.pertinent_table)
            st.divider()
            
            result  = geo.process_geotechnical_data( st.session_state.pertinent_table )

            st.dataframe( st.session_state.pertinent_table[result.output.names_of_columns_used_for_extraction] )
            
    with interface_container:
        display_main_interface()

# ==============================================================================
# 3. MAIN APPLICATION
# ==============================================================================

def landing_page():
    """Main application entry point."""
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
            # TEMPORARY 
            # Offer download
            json_filename = "etude_de_sol.json"
            json_str = json.dumps(st.session_state['aryn_result'] , indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Télécharger le résultat JSON",
                data=json_str,
                file_name=json_filename,
                mime="application/json"
            )

if __name__ == "__main__":
    landing_page()
