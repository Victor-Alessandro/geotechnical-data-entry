import streamlit as st
import geopandas as gpd
import pandas as pd
import cv2


from utils import ndc

def initialize_session_state():
    """Initialize the application with dependency injection"""
    data_manager = ndc.DataManager(st.session_state)
    
    defaults = data_manager.load_defaults()
    data_manager.initialize_session_state(defaults)
    
    return data_manager

def render_file_upload_section():
    """Render the initial file upload section"""
    col_nav1, col_nav2 = st.columns(2, vertical_alignment="center")
    with col_nav1:
        st.header("📁 Upload des Documents Requis")
    with col_nav2:
        if st.button("Détails du document", type="primary", use_container_width=True):
            if st.session_state.get('calculs_excel') and st.session_state.get('plan_masse'):
                # Set a flag to indicate user wants to proceed
                st.session_state.proceed_to_main = True
                st.rerun()
            else:
                st.toast("⚠️ Assurez-vous que le calcul de la portance ainsi que le plan de masse soient téléversés.")
   
    col1_docs, col2_docs, col3_docs = st.columns(3)
    
    with col1_docs:
        excel = st.file_uploader(
            "1. Téléversez le calcul de la portance (.xlsm, .xlsx)", 
            type=['xlsm', 'xlsx'], 
            key="calculs_excel_uploader"
        )
        if excel:
            required_sheets = ['Dimensionnement Général', 'Hypothèses'] 
            if ndc.validate_excel_file(excel, required_sheets):
                st.session_state.calculs_excel = excel 
                st.toast("Fichier Excel de calcul validé!", icon="✅")
                
    with col2_docs:
        st.session_state.plan_masse = st.file_uploader(
            "2. Téléversez le plan de masse (.jpg, .png, .pdf)", 
            type=['jpg', 'jpeg', 'png', 'pdf'], 
            key="plan_masse_uploader"
        )
        if st.session_state.plan_masse:
            try:
                st.session_state.plan_masse.seek(0) 
                if st.session_state.plan_masse.type.startswith('image'):
                    st.image(st.session_state.plan_masse, caption="Aperçu Plan de masse", width=300)
                elif st.session_state.plan_masse.type == 'application/pdf':
                    converted_plan = ndc.convert_pdf_to_image(st.session_state.plan_masse)
                    st.image(converted_plan, caption="Aperçu Plan de masse (PDF converti)", width=300)
                st.session_state.plan_masse.seek(0) 
            except Exception as e:
                st.error(f"Erreur d'affichage du plan de masse: {e}")
    
    with col3_docs:
        additional_excel = st.file_uploader(
            "3. Optionnel: Effort Horizontal (.xlsm, .xlsx)", 
            type=['xlsm', 'xlsx'], 
            key="additional_excel_uploader"
        )
        if additional_excel:
            required_sheets = ['Hypothèses', 'Calcul Horiz', 'Cas Multiples'] 
            if ndc.validate_excel_file(additional_excel, required_sheets):
                st.session_state.horiz_excel = additional_excel
                st.toast("Fichier Excel supplémentaire validé!", icon="✅")



          
def render_page_garde_section():
    """Render the page de garde section"""
    st.header("📄 Page de Garde")
    
    col1_pg, col2_pg, col3_pg = st.columns(3)
    
    with col1_pg:
        st.session_state.nom = st.text_input("NOM Prénom", value=st.session_state.nom)
        st.session_state.soustitre = st.text_input("Sous-titre", value=st.session_state.soustitre)
    
    with col2_pg:
        ndc.handle_commune_input() 
        techniques = list(ndc.TECHNIQUES.keys())
        current_technique = st.session_state.technique
        technique_idx = techniques.index(current_technique) if current_technique in techniques else 0
        st.session_state.technique = st.selectbox("Technique de réalisation", techniques, index=technique_idx)
    
    with col3_pg:
        affaire_input = st.text_input("Nº Affaire", value=st.session_state.affaire, max_chars=5)
        if affaire_input != st.session_state.affaire:
            if not affaire_input or ndc.FormValidator.validate_affaire_number(affaire_input):
                st.session_state.affaire = affaire_input
            else:
                st.error("Le numéro d'affaire doit être composé de 5 chiffres.")
    
    with st.expander("Champs optionnels (Maîtrise d'ouvrage, etc.)"):
        col1_opt, col2_opt, col3_opt = st.columns(3)
        with col1_opt:
            st.session_state.maitre = st.text_input("Maître d'ouvrage", value=st.session_state.maitre)
        with col2_opt:
            st.session_state.promoteur = st.text_input("Promoteur / Architecte", value=st.session_state.promoteur)
        with col3_opt:
            st.session_state.geotechnicien = st.text_input("Bureau d'études Géotechniques", value=st.session_state.geotechnicien)


def render_pieux_dimensioning():
    """Render technical information about the poles being built"""
    # I'll be treating this as a read only item
    if not st.session_state.display_dataframes:
        st.session_state.display_dataframes = ndc.instantiate_dataframes(st.session_state)

    st.header("📐 Détails des Pieux")
    
    col1, col2 = st.columns([2,1])
    
    with col1:
        beton_types = list(ndc.BETON_TYPES.keys())
        current_beton_type = st.session_state.beton_type
        beton_idx = beton_types.index(current_beton_type) if current_beton_type in beton_types else 0
        st.session_state.beton_type = st.selectbox("Type de béton", beton_types, index=beton_idx)

        # clunky
        current_diameter_option = st.session_state.diameter_option
        diameter_idx = ndc.DIAMETER_OPTIONS.index(current_diameter_option) if current_diameter_option in ndc.DIAMETER_OPTIONS else 0
        st.session_state.diameter_option = st.selectbox("Option de diamètre", ndc.DIAMETER_OPTIONS, index=diameter_idx)

        def available_columns( name: str ):
          if tables_dict := st.session_state.display_dataframes:
              if name in tables_dict.keys():
                  verif_df = tables_dict[name]
                  if not verif_df.empty:
                      return list(verif_df.columns)

        multiselect_options = available_columns( "verifications" )

        # I could alias these names, but not doing so seems simpler more transparent. 
        synthese_defaults = [item for item in st.session_state.selection_synthese if item in multiselect_options]
        st.session_state.selection_synthese = st.multiselect('Colonnes du Tableau de Synthèse', multiselect_options, synthese_defaults)

    with col2:
        acier_types = list(ndc.ACIER_TYPES.keys())
        current_acier_type = st.session_state.type_acier
        acier_idx = acier_types.index(current_acier_type) if current_acier_type in acier_types else 0
        st.session_state.type_acier = st.selectbox("Type d'Acier", acier_types, index=acier_idx)
        
        st.markdown(":small[Documents de Marché]", help="Vous pouvez ajouter plusieurs lignes. Ce champ correspond aux documents de marché des NDC.")
        
        if (not isinstance(st.session_state.get("documents"), pd.DataFrame) or 
            "Documents de Marché" not in st.session_state.documents.columns):
            st.session_state.documents = pd.DataFrame({"Documents de Marché": st.session_state.get("documents_marche", [""])})
        edited_df = st.data_editor(
            st.session_state.documents,
            num_rows="dynamic",
            use_container_width=True,
            key="doc_editor",
        )
        
        if "Documents de Marché" in edited_df.columns:
            st.session_state.documents_marche = edited_df["Documents de Marché"].dropna().astype(str).tolist()

    resultat_defaults = [item for item in st.session_state.selection_resultats if item in multiselect_options]
    st.session_state.selection_resultats = st.multiselect('Colonnes du Tableau de Résultats', multiselect_options, resultat_defaults)

    if st.session_state.diameter_option == "Plusieurs diamètres":
        #too many global variables.

        st.divider()
        
        diameters_df = ndc.init_select_column(st.session_state.display_dataframes["diameters"][st.session_state.selection_diameters])
        diameter_options = available_columns( "diameters" )

        diameter_defaults = [item for item in st.session_state.selection_diameters if item in diameter_options]
        st.session_state.selection_diameters = st.multiselect('Colonnes du Tableau de type de Béton', diameter_options, diameter_defaults)
    
        st.session_state.selected_dataframe = ndc.dataframe_with_selections( diameters_df )

        if not st.session_state.selected_dataframe.empty:
            st.write("🎯 Selection pour le béton:")
            st.dataframe(st.session_state.selected_dataframe, use_container_width=True, hide_index=True)

    if st.session_state.horiz_excel:
        st.divider()

        # I could alias these names, but not doing so seems simpler, more transparent. 
        horiz_defaults = [item for item in st.session_state.selection_horizontal if item in multiselect_options]
        st.session_state.selection_horizontal = st.multiselect('Colonnes du Tableau des Efforts Horizontales', multiselect_options, horiz_defaults)

    #st.session_state.display_dataframes['geodata']
    #colors = ['FF5733', '33C1FF', '7DFF33', 'F0E68C', '8A2BE2']

def render_sismicite_section():
    """Render the sismicité section"""
    st.header("🗺️ Analyse Sismique")

    batiment_options = ["I", "II avec application possible des PS-MI / CP-MI", "II", "III", "IV"]
    current_batiment_type = st.session_state.type_batiment
    batiment_idx = batiment_options.index(current_batiment_type) if current_batiment_type in batiment_options else 0

    st.session_state.type_batiment = st.selectbox(
        "Catégorie d'importance du bâtiment",
        batiment_options,
        index=batiment_idx,
        help="""Selon l'arrêté du 22 octobre 2010 relatif à la classification et aux règles de construction parasismique :
        - **Catégorie I**: Bâtiments dont la défaillance présente un risque minime pour les personnes ou l'activité économique.
        - **Catégorie II**: Bâtiments courants. (Ex: logements individuels, bureaux non ERP de moins de 300 personnes)
        - **Catégorie III**: Bâtiments dont la protection est primordiale. (Ex: ERP importants, établissements scolaires, centres de santé)
        - **Catégorie IV**: Bâtiments stratégiques et indispensables à la gestion de crise. (Ex: hôpitaux vitaux, centres de secours, communications)
        """
    )
    
    if st.session_state.get("commune") and st.session_state.get("sismicite"):
        st.info(f"La commune **{st.session_state.commune.split(' (')[0]}** est en **{st.session_state.sismicite}**.")
    else:
        st.warning("Veuillez sélectionner une commune pour afficher sa zone de sismicité.")

def render_seismic_map():
    """Render seismic map section"""
    shapefile_path = "./resources/france_zonage_sismique/France_zonage_sismique.shp"

    try:
        data = gpd.read_file(shapefile_path) 
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier: {e}")
        return None

    if st.session_state.selected_gdf is not None and data is not None:
        dept_id = st.session_state.selected_departement
        commune_name = st.session_state.selected_name
        dept_gdf = data[data['departemen'] == dept_id].copy()

        if not dept_gdf.empty:
            buf = ndc.produce_departement_map(
                gdf_commune=st.session_state.selected_gdf,
                gdf_departement=dept_gdf,
                selected_commune_name=commune_name,
                departement_identifier=dept_id,
                base_crs=data.crs.to_string()
            )

            st.session_state.commune_map_buffer = buf 
            if st.button(f"Afficher la carte de sismicité pour {commune_name} (Département: {dept_id})", key="show_seismic_map_btn"):
                if hasattr(st.session_state, 'commune_map_buffer') and st.session_state.commune_map_buffer:
                    st.session_state.commune_map_buffer.seek(0)
                    st.image(st.session_state.commune_map_buffer, caption=f"Carte de sismicité pour {commune_name} (Département: {dept_id})")
                else:
                    st.info("La carte de sismicité n'est pas encore disponible ou la commune n'est pas sélectionnée.")
        else:
            st.toast(f"Aucune donnée cartographique pour le département {dept_id}.", icon="⚠️")
    else:
        st.info("Entrez un nom de commune valide pour que la carte de sismicité puisse être générée.")

def render_norm_table_highlight():
    """Render highlighted norm table section"""
    if st.session_state.type_batiment and st.session_state.sismicite:
        st.session_state.highlighted_image = ndc.highlight_coordinates()

        if st.button("Catégorie du batîment et norme sismique", key="show_norm_table_cell_btn"):
            if hasattr(st.session_state, 'highlighted_image') and st.session_state.highlighted_image is not None:
                st.image(cv2.cvtColor(st.session_state.highlighted_image, cv2.COLOR_BGR2RGB),
                           caption=f"Cellule surlignée (Sismicité: {st.session_state.sismicite}, Catégorie: {st.session_state.type_batiment})")
            else:
                st.info("L'image surlignée n'est pas disponible. Vérifiez les sélections de catégorie/sismicité ou le fichier image de fond.")
    else:
        st.info("Sélectionnez une catégorie de bâtiment et une commune pour générer/afficher la surbrillance sur le tableau sismique.")
        if 'highlighted_image' in st.session_state:
            st.session_state.highlighted_image = None

def render_data_entry_form():
    """Render the main data entry form after files are uploaded"""      
    render_page_garde_section()
    st.divider()

    render_pieux_dimensioning()
    st.divider()

    render_sismicite_section()
    render_seismic_map()
    render_norm_table_highlight()
    st.divider()

def main():
    """Main application function for NDC data input"""
    st.set_page_config(
        page_title="Générateur NDC Pieux", 
        page_icon="🏗️", 
        layout="wide",
    )
    
    st.title("🏗️ Générateur de Notes de Calcul")
    
    initialize_session_state()
    
    # Check if user has clicked proceed button AND required files are present
    should_show_main = (
        st.session_state.proceed_to_main and 
        st.session_state.calculs_excel and 
        st.session_state.plan_masse
    )
    
    if not should_show_main:
        render_file_upload_section()
    else:
        render_data_entry_form()
        col1, col2 = st.columns(2, vertical_alignment='center')
        # the visibility of st.page_link sucks
        with col1:
            st.page_link('🔽 Téléchargements.py', label='**🔽 Téléchargements**')
        with col2:

           # modifying this might be needed but I'm not sure because of the global scope of session state
           # namely, When defaults are loaded again after a new document gets uploaded do they fully override the previous state variables ?
           # or are side effects left behind from the last use ?
           if st.button("🔄 Modifier les fichiers", key="back_to_upload", type='tertiary'):

               st.session_state.calculs_excel = None
               st.session_state.plan_masse = None  # Fixed typo: was plan_de_masse
               st.session_state.horiz_excel = None
               st.session_state.proceed_to_main = False
               st.rerun()
    
        st.success("✅ Données prêtes pour la génération de la Note de Calcul.")


if __name__ == "__main__":
    main()
