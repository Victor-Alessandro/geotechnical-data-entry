import streamlit as st
import pandas as pd
import tempfile
import fitz  # PyMuPDF
import os

from typing import  List
from streamlit_extras.pdf_viewer import pdf_viewer
from pydantic import create_model
from pydantic_ai import Agent 

from utils.reusables import strip_pdf
from utils import ddc

st.set_page_config(
    page_title="Descentes de Charges",
    page_icon=":classical_building:",
    layout="wide"
)


def initialize_session_state():
    """Initialize all required session state variables"""
    if 'ddc_uploaded' not in st.session_state:
        st.session_state.ddc_uploaded = False
    if 'ddc_tables' not in st.session_state:
        st.session_state.ddc_tables = None
    if 'selected_ddc_table' not in st.session_state:
        st.session_state.selected_ddc_table = None
    if 'formatted_ddc_table' not in st.session_state:
        st.session_state.formatted_ddc_table = None
    if 'crop_params' not in st.session_state:
        st.session_state.crop_params = {'x': 0, 'y': 0, 'width': 100, 'height': 100}
    if 'final_ddc_data' not in st.session_state:
        st.session_state.final_ddc_data = None
    if 'page_info' not in st.session_state:
        st.session_state.page_info = {}
    if 'all_extracted_tables' not in st.session_state:
        st.session_state.all_extracted_tables = {}
    if 'max_workers' not in st.session_state:
        st.session_state.max_workers = 2
    if 'formatted_table' not in st.session_state:
        st.session_state.formatted_table = None
    if 'final_dataframe' not in st.session_state:
        st.session_state.final_dataframe = None

def main():
    st.title(":classical_building: Descentes de Charges")
    
    initialize_session_state()

    # File upload section
    if not st.session_state.ddc_uploaded:
        # This will be the generic landing page, I should create a function for it
        st.header("📁 Upload du Document")
        
        charges = st.file_uploader(
            "Choisir le fichier DDC",
            type=['pdf'],
            key="charges"
        )
        
        if charges:           
            if charges.type == "application/pdf" or charges.type.startswith("image/"):
                st.session_state.ddc_uploaded = True

                # unnecessary, really
                st.session_state.original_ddc = charges

                st.session_state.ddc_file = strip_pdf(charges)
                st.success("Document ouvert avec succès!")
                st.rerun()
    
    else:
        st.header("🔧 Traitement DDC")
                
        col1, col2 = st.columns([2, 1])
        with col1:
            
            pdf_viewer(st.session_state.ddc_file)

        with col2:
            st.info(f"📄 Fichier: {st.session_state.original_ddc.name}")

            if st.button("🔄 Changer de fichier"):
                st.session_state.ddc_uploaded = False
                st.session_state.ddc_tables = None
                st.session_state.all_extracted_tables = {}
                st.rerun()

            st.divider()
            
            with st.expander("⚙️ Paramètres d'Extraction", expanded=False):
                max_workers = st.slider(
                    "Nombre de workers parallèles", 
                    min_value=1, 
                    max_value=4, 
                    value=st.session_state.max_workers,
                    help="Plus de workers = plus de pages traités en parallèle, mais plus de mémoire utilisée",
                    key="workers_slider"
                )
                st.session_state.max_workers = max_workers
                
               
                if st.session_state.original_ddc.type == "application/pdf":
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(st.session_state.ddc_file)
                        pdf_path = tmp_file.name
                    
                    try:
                        pdf = fitz.open(pdf_path)
                        num_pages = len(pdf)
                        pdf.close()
                        
                        st.write(f"Document PDF avec **{num_pages} pages**")
                        
                        page_selection = st.selectbox(
                            "Sélectionnez les pages à traiter",
                            ["Toutes les pages", "Première page uniquement", "Range personnalisée"],
                            key="page_selection"
                        )
                        
                        if page_selection == "Range personnalisée":
                            page_range = st.text_input(
                                "Entrez les numéros de page (séparés par des virgules, indexés à partir de 0)", 
                                "0",
                                key="page_range"
                            )
                            try:
                                page_nums = [int(p.strip()) for p in page_range.split(",")]
                                
                                page_nums = [p for p in page_nums if 0 <= p < num_pages]
                                if not page_nums:
                                    st.warning("Numéros de page invalides. Utilisation de la première page.")
                                    page_nums = [0]
                            except:
                                st.warning("Entrée invalide. Utilisation de la première page.")
                                page_nums = [0]
                        elif page_selection == "Toutes les pages":
                            page_nums = list(range(num_pages))
                        else:  
                            page_nums = [0]
                        
                        extraction_method = st.radio(
                            "Méthode d'extraction de tableau",
                            ["Séparée par Lignes", "Séparée par Espaces"],
                            index=0,
                            horizontal=True,
                            help="Comment l'algorithme s'attend à ce que les cellules des tableaux soient séparés dans le document.",
                            key="extraction_method"
                        )
                        
                        dpi = st.slider(
                            "Densité de pixels de l'image", 
                            min_value=72, 
                            max_value=400, 
                            value=120, 
                            step=72,
                            key="dpi_slider"
                        )
                        
                        st.session_state.extraction_params = {
                            'page_nums': page_nums,
                            'extraction_method': extraction_method,
                            'dpi': dpi,
                            'pdf_path': pdf_path
                        }
                        
                    except Exception as e:
                        st.error(f"Erreur lors de l'ouverture du PDF: {e}")
                        try:
                            os.unlink(pdf_path)
                        except:
                            pass
                        return
                    
                else:
                    st.info("Fichier image détecté - les paramètres PDF ne sont pas applicables")

            if st.button("🔍 Extraire Tableaux", key="extract_ddc_tables", type='primary'):
                if st.session_state.original_ddc.type == "application/pdf" and hasattr(st.session_state, 'extraction_params'):
                    with st.spinner("Extraction des tableaux en cours..."):
                        params = st.session_state.extraction_params
                        
                        all_tables, camelot_tables, page_info = ddc.extract_tables_from_pdf_parallel(
                            params['pdf_path'],
                            params['page_nums'],
                            params['extraction_method'],
                            st.session_state.get('max_workers', 2)
                        )
                        
                        st.session_state.all_extracted_tables = all_tables
                        st.session_state.page_info = page_info
                        st.session_state.camelot_tables = camelot_tables
                        
                        total_tables = sum(len(tables) for tables in all_tables.values())
                        st.success(f"✅ {total_tables} tableaux extraits de {len(params['page_nums'])} pages!")
                        
                        try:
                            os.unlink(params['pdf_path'])
                        except:
                            pass
                else:
                    st.error("Veuillez d'abord configurer les paramètres d'extraction pour un fichier PDF.")
            
        st.divider()
        
        if st.session_state.all_extracted_tables:
            
            st.subheader("📊 Tableaux Extraits - Éditables")
        
            table_titles = [table_info['table_title']
                           for _, list_of_tables in st.session_state.all_extracted_tables.items()
                           for table_info in list_of_tables]           

            selected_table_name = st.selectbox("Tableau Sélectionné", table_titles)
              
            for page_num, tables in st.session_state.all_extracted_tables.items():
                page_info = st.session_state.page_info.get(page_num, "Format inconnu")

                for table_info in tables:
                    expanded_bool = True if table_info['table_title'] == selected_table_name else False

                    if expanded_bool:
                        st.session_state.selected_ddc_table = table_info['dataframe']
                        
                    with st.expander(table_info['table_title'], expanded=expanded_bool):
                        # Display table info
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            st.info(f"**Page d'origine:** {page_num + 1}")
                        with col_info2:
                            st.info(f"**Format:** {table_info['paper_size']}")
                        
                        # Editable data editor
                        editor_key = f"table_editor_{table_info['global_index']}"
                        edited_df = st.data_editor(
                            table_info['dataframe'],
                            key=editor_key,
                            use_container_width=True,
                            num_rows="dynamic",
                            height=min(400, max(200, len(table_info['dataframe']) * 35 + 100))
                        )                        
                        table_info['dataframe'] = edited_df # Update the table info with edited data 
            
            st.divider()

            # UI inputs
            os.environ['GEMINI_API_KEY'] = st.secrets['GOOGLE_API_KEY']
            with st.expander("Requête LLM", expanded=True):
                
                column_names = st.multiselect(
                    label="Colonnes à construire:",
                    options=[
                        'Q (Charges d\'Exploitation)', 'G (Charges Permanentes)',
                        'V (Charges de Vent)', 'W (Charges de Vent)',
                        'N (Charges de Neige)', 'S (Charges de Neige)',
                        'ELS (États Limites en Service)', 'ELU (États Limite Ultimes)',
                        'Sismique', 'S+', 'S-', 'Hsis', 'Vsis', 'Newmark', 'Fx', 'Fy', 'Fz',
                        'Moments', 'Mx', 'My', 'Mz', 'Vz', 'Vg', 'Vq', 'Vw', 'Vn', 'Hg','Hz','Hq','Hw', 'Hn', 'Verticale', 'Horizontale'                     
                    ],
                    default=['Q (Charges d\'Exploitation)', 'G (Charges Permanentes)'],
                    key='1',
                    accept_new_options=True
                )

                complement_sys_prompt ="\nRépondez uniquement en JSON conforme au modèle DictFormat. Les keys du dictionnaire seront les noms des colonnes et son contenu les listes de valeurs que tu ira selectionner."
                
                sys_prompt = st.text_area(
                    "La requête",
                    value=(
                        f"Filtrez le contenu du tableau afin de garder uniquement les valeurs numériques correspondant aux colonnes suivantes : {column_names}.\n"
                        "Ne modifiez en aucun cas le contenu à l'intérieur des cellules."
                    ),
                    help=f"Texte du prompt système pour guider la réponse du modèle.\nContinue par:{complement_sys_prompt} "
                )

                
                sys_prompt = sys_prompt + complement_sys_prompt

                df = st.session_state.selected_ddc_table.apply( pd.to_numeric, errors='ignore')

                if "Ne modifiez en aucun cas le contenu" in sys_prompt:
                    df =  ddc.clean_large_table(df)
                
                prompt = f"Faites attention à l'extraction de tous les valeurs des colonnes demandés:\n{df.to_markdown()}"

                def format_dataframe( sys_prompt, prompt, dataframe):
                    fields = {col: (List[float], ...) for col in column_names}
                    DictFormat = create_model("DictFormat", **fields)
                    
                    formatting_agent = Agent[None, dict](
                        'gemini-2.0-flash',
                        output_type=DictFormat,
                        sys_prompt=sys_prompt,
                    )
            
                    result = formatting_agent.run_sync( prompt )

                    return pd.DataFrame(result.output.dict())

                if st.button("Formater le tableau\n\rSélectionné"):
                    if st.session_state.all_extracted_tables:
                        try:
                            st.session_state.final_dataframe = st.data_editor( format_dataframe( sys_prompt, prompt, df ))
                            st.success("Tableaux formatés et prêts pour édition!")

                        except Exception as e:
                            st.error(f"Un erreur s'est produit, c'est probable que l'IA du serveur n'aït pas répondu.\nerreur accusé: {e}")
                    else:
                        st.error("Aucun tableau extrait. Extrayez d'abord les tableaux.")


            
        st.sidebar.subheader(":classical_building:  Statut")       
           
        if st.session_state.all_extracted_tables:
            total_tables = sum(len(tables) for tables in st.session_state.all_extracted_tables.values())
            st.sidebar.success(f"✅ {total_tables} tableaux extraits")
            
            # Show page breakdown
            st.sidebar.markdown("**Répartition par page:**")
            for page_num, tables in st.session_state.all_extracted_tables.items():
                st.sidebar.text(f"Page {page_num + 1}: {len(tables)} tableaux")
        else:
            st.sidebar.info("❌ Aucun tableau extrait")

if __name__ == "__main__":
    main()
