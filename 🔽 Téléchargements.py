
import streamlit as st
import pandas as pd
import io
from typing import Optional
from utils import ndc  # Assuming 'utils.ndc' is accessible
from utils.reusables import initialize_session_state

# Page configuration
st.set_page_config(
    page_title="Analyse Géotechnique et Fichiers",
    page_icon="🏗️",
    layout="wide",
)

# --- Utility functions ---

def generate_excel_files(geo_data: Optional[pd.DataFrame], ddc_data: Optional[pd.DataFrame]) -> tuple:
    """Generate Excel files for download"""
    geo_excel, ddc_excel = None, None
    if geo_data is not None and not geo_data.empty:
        geo_buffer = io.BytesIO()
        geo_data.to_excel(geo_buffer, index=False)
        geo_excel = geo_buffer.getvalue()
    if ddc_data is not None and not ddc_data.empty:
        ddc_buffer = io.BytesIO()
        ddc_data.to_excel(ddc_buffer, index=False)
        ddc_excel = ddc_buffer.getvalue()
    return geo_excel, ddc_excel

# --- Main Page Logic ---

def main():

    KEYS_TO_INIT = {
        'final_geo_data': None,
        'final_ddc_data': None,
        'generated_docx': None,
        'calculs_excel': None,
        'plan_masse': None,
        'generated_image_paths': None,
        'display_dataframes': None,
    }

    initialize_session_state(KEYS_TO_INIT)

    st.title(" 🔽 Téléchargement des Fichiers")
    st.write("Utilisez les pages dédiées pour traiter vos documents, puis revenez ici pour générer et télécharger les fichiers finaux.")

    st.divider()

    st.header("📊 Fichiers d'Analyse")
    st.markdown("Ces fichiers contiennent les données brutes extraites des pages 'Étude de Sol' et 'Descentes de Charges'.")
    
    col1, col2 = st.columns(2)
    
    geo_excel, ddc_excel = generate_excel_files(
        st.session_state.final_geo_data,
        st.session_state.final_ddc_data
    )
    
    with col1:
        st.subheader("Calculs Excel")
        if st.session_state.final_geo_data is not None:
            st.dataframe(st.session_state.final_geo_data, use_container_width=True)
            if geo_excel:
                st.download_button(
                    label="📥 Télécharger Excel Géotechnique",
                    data=geo_excel,
                    file_name="donnees_geotechniques.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.info("Il faut d'abord que l'étude de sol soit chargé et ses données extraits.")
    
    with col2:
        st.subheader("Descentes de Charges (DDC)")
        if st.session_state.final_ddc_data is not None:
            st.dataframe(st.session_state.final_ddc_data, use_container_width=True)
            if ddc_excel:
                st.download_button(
                    label="📥 Télécharger Excel DDC",
                    data=ddc_excel,
                    file_name="donnees_ddc.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.info("Il faut d'abord que le PDF avec les DDC soit chargé et le tableau extrait.")

    # --- Section for Note de Calcul (NDC) ---
    st.divider()
    st.header("📝 Note de Calcul")

    if st.session_state.get('calculs_excel') and st.session_state.get('plan_masse'):
        st.success("Les fichiers requis (Calcul de portance et Plan de masse) sont chargés.")

        st.subheader("Génération du Document Final")
        
        if st.button("🔧 Générer la Note de Calcul", key="generate_doc_button_fichiers", help="Cliquez pour lancer la génération du document Word."):
            validation_errors = ndc.FormValidator.validate_required_fields(st.session_state)
            if validation_errors:
                for error in validation_errors:
                    st.error(error)
                st.session_state.generated_docx = None
            else:
                with st.spinner("Génération du document en cours... Veuillez patienter."):
                    try:
                        # after the document has been generated I should clear at least the files from the respective
                        # pages. At least related with the diameters table there is a side effect, and its unlikely to be
                        # the only one.
                        
                        template_path ="./resources/"                        
                        if st.session_state.horiz_excel:
                            if st.session_state.diameter_option == "Plusieurs diamètres":
                                template_path = template_path + "aciers2.docx"
                            else:
                                template_path = template_path + "aciers.docx"
                        else:
                            if st.session_state.diameter_option == "Plusieurs diamètres":
                                template_path = template_path + "normal2.docx"
                            else:
                                template_path = template_path + "normal.docx"
                         
                        doc_generator = ndc.DocumentGenerator( template_path )
                        context = ndc.build_document_context(st.session_state)
                        
                        st.session_state.highlighted_image = ndc.generate_highlighted_seismic_table_image_data(
                            st.session_state.sismicite,
                            st.session_state.type_batiment
                        )
                        st.session_state.generated_image_paths = doc_generator.prepare_images(st.session_state) # For download buttons

                        # Prepare images and dataframes for display and inclusion
                        st.session_state.generated_image_paths = doc_generator.prepare_images(st.session_state)

                        # Generate the document
                        docx_file = doc_generator.generate_document(context)
                        doc_generator.cleanup()

                        if docx_file:
                            st.session_state.generated_docx = docx_file
                            st.toast("Document généré avec succès!", icon="🎉")
                        else:
                            st.session_state.generated_docx = None
                            st.error("Erreur: Le document n'a pas pu être généré.")
                    except Exception as e:
                        st.session_state.generated_docx = None
                        st.error(f"Erreur critique lors de la génération: {str(e)}")

        if st.session_state.generated_docx:
            download_file_name = f"NDC_{st.session_state.get('affaire', '00000')}_{st.session_state.get('commune', 'Commune').split(' (')[0].replace(' ','_')}.docx"
            st.download_button(
                label="📥 Télécharger la Note de Calcul (.docx)",
                data=st.session_state.generated_docx,
                file_name=download_file_name,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="download_docx_btn_fichiers"
            )

    else:
        st.info(
            """
            Il faut d'abord que le calcul de la portance et une image du plan de masse soient chargés.
            """
        )

if __name__ == "__main__":
    main()
