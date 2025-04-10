# import os
# import logging
# import streamlit as st
# import traceback
# import pandas as pd
# from datetime import datetime
# import hashlib
# import cv2
# import numpy as np
# import time

# # Import modules
# from preprocess import read_image, extract_id_card, save_image
# from ocr_engine import extract_text
# from postprocess import extract_information, extract_information1
# from face_verification import detect_and_extract_face, deepface_face_comparison, get_face_embeddings
# from sql_connection import insert_records, fetch_records, check_duplicacy, insert_records_aadhar, fetch_records_aadhar, check_duplicacy_aadhar
# from selfie_capture import integrate_with_streamlit, verify_face_with_id  # Import the new module

# # Set up logging
# def setup_logging():
#     logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
#     log_dir = "logs"
#     os.makedirs(log_dir, exist_ok=True)
#     logging.basicConfig(
#         filename=os.path.join(log_dir, "ekyc_logs.log"), 
#         level=logging.INFO, 
#         format=logging_str, 
#         filemode="a"
#     )

# # Enhanced page styling
# def set_page_style():
#     st.set_page_config(
#         page_title="E-KYC Registration using OCR and Computer Vision",
#         page_icon=":id:",
#         layout="wide",
#         initial_sidebar_state="expanded"
#     )
    
#     st.markdown("""
#     <style>
#     .reportview-container {
#         background: linear-gradient(to right, #f0f2f6, #e6e9f0);
#     }
#     .sidebar .sidebar-content {
#         background-color: #ffffff;
#         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#     }
#     .stButton>button {
#         background-color: #4CAF50;
#         color: white;
#         border-radius: 5px;
#         transition: all 0.3s ease;
#     }
#     .stButton>button:hover {
#         background-color: #45a049;
#         transform: scale(1.05);
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # Hash sensitive information
# def hash_id(id_value):
#     try:
#         hash_object = hashlib.sha256(str(id_value).encode())
#         return hash_object.hexdigest()
#     except Exception as e:
#         logging.error(f"Error hashing ID: {e}")
#         return id_value

# def main_content(image_file, option, use_webcam=False):
#     try:
#         # Input validation for ID card
#         if not image_file:
#             st.warning("Please upload ID card image.")
#             return

#         # Read ID card image
#         image = read_image(image_file, is_uploaded=True)
        
#         if image is None:
#             st.error("Unable to read uploaded image. Please try again.")
#             return

#         # Extract ID card region of interest
#         image_roi, _ = extract_id_card(image)
        
#         # Face image acquisition
#         face_image_path = None
        
#         if use_webcam:
#             with st.spinner("Capturing selfie..."):
#                 face_image_path = integrate_with_streamlit()
                
#             if not face_image_path:
#                 st.error("Selfie capture failed. Please try again.")
#                 return
                
#             st.success("Selfie captured successfully!")
#         else:
#             # If not using webcam, get face image from upload
#             face_image_file = st.session_state.get('face_image_file')
#             if not face_image_file:
#                 st.warning("Please upload face image or use webcam.")
#                 return
                
#             face_image = read_image(face_image_file, is_uploaded=True)
#             if face_image is None:
#                 st.error("Unable to read uploaded face image. Please try again.")
#                 return
                
#             face_image_path = save_image(face_image, "face_image.jpg", path="data/02_intermediate_data")
        
#         # Extract face from ID card for comparison
#         face_image_path_id = detect_and_extract_face(img=image_roi)
        
#         if face_image_path_id is None:
#             st.error("No face detected in the ID card. Please upload a clearer image.")
#             return
        
#         # Face verification
#         is_face_verified = deepface_face_comparison(
#             image1_path=face_image_path, 
#             image2_path=face_image_path_id
#         )

#         # Text extraction
#         extracted_text = extract_text(image_roi)
        
#         # Extract information based on ID type
#         text_info = (
#             extract_information(extracted_text) if option == "PAN" 
#             else extract_information1(extracted_text)
#         )

#         # Display extracted information
#         st.subheader("Extracted Information")
#         st.json(text_info)  # Display raw extracted info
        
#         # Display verification status
#         verification_col1, verification_col2 = st.columns(2)
#         with verification_col1:
#             st.write(f"Face Verification Status: {'✅ Verified' if is_face_verified else '❌ Not Verified'}")
#         with verification_col2:
#             if use_webcam:
#                 st.write("Selfie Capture: ✅ Completed")
        
#         # If face verification failed, stop processing
#         if not is_face_verified:
#             st.error("Face verification failed. The face in the ID card does not match the provided face image.")
#             return
        
#         # Hash ID
#         original_id = text_info['ID']
#         text_info['ID'] = hash_id(text_info['ID'])

#         # Duplicate check
#         is_duplicate = (
#             check_duplicacy(text_info) if option == "PAN" 
#             else check_duplicacy_aadhar(text_info)
#         )

#         if is_duplicate:
#             st.warning(f"User already registered with ID {original_id}")
#             return

#         # Process and insert record
#         try:
#             text_info['DOB'] = (
#                 text_info['DOB'].strftime('%Y-%m-%d') 
#                 if isinstance(text_info['DOB'], datetime) 
#                 else str(text_info['DOB'])
#             )
#         except Exception as date_err:
#             logging.warning(f"Date formatting error: {date_err}")

#         # Get face embeddings
#         text_info['Embedding'] = get_face_embeddings(face_image_path)

#         # Insert records
#         if option == "PAN":
#             insert_records(text_info)
#         else:
#             insert_records_aadhar(text_info)

#         st.success("Registration Successful!")

#     except Exception as e:
#         st.error(f"An error occurred: {str(e)}")
#         logging.error(f"Error in main_content: {traceback.format_exc()}")
        
# # Main Streamlit app
# def main():
#     setup_logging()
#     set_page_style()

#     # Initialize session state for storing face image
#     if 'face_image_file' not in st.session_state:
#         st.session_state['face_image_file'] = None

#     st.title("E-KYC Registration using OCR and Computer Vision")
#     st.write("Developed by 22MID0081 and 22MID0108")
    
#     # Sidebar for ID type selection
#     option = st.sidebar.radio(
#         "Select ID Card Type", 
#         ["PAN", "AADHAR"], 
#         index=0
#     )

#     # File uploader for ID card
#     image_file = st.file_uploader(
#         f"Upload {option} Card Image", 
#         type=['png', 'jpg', 'jpeg']
#     )
    
#     # Face image acquisition method selection
#     face_acquisition_method = st.radio(
#         "How would you like to provide your face image?",
#         ["Upload Image", "Use Webcam (Capture Selfie)"]
#     )
    
#     use_webcam = face_acquisition_method == "Use Webcam (Capture Selfie)"
    
#     if not use_webcam:
#         # File uploader for face image
#         face_image_file = st.file_uploader(
#             "Upload Face Image", 
#             type=['png', 'jpg', 'jpeg']
#         )
#         st.session_state['face_image_file'] = face_image_file

#     # Process registration
#     if st.button("Register"):
#         main_content(image_file, option, use_webcam)

# if __name__ == "__main__":
#     main()


import pandas as pd
from datetime import datetime
import re
import logging

def extract_information(data_string):
    try:
        # Preprocess the input
        updated_data_string = data_string.replace(".", "")
        words = [word.strip() for word in updated_data_string.split("|") if len(word.strip()) > 2]
        
        # Initialize extracted info with default values
        extracted_info = {
            "ID": "",
            "Name": "",
            "Father's Name": "",
            "DOB": "",
            "ID Type": "PAN"
        }

        # Try to extract information with fallback mechanisms
        try:
            # Name extraction with multiple fallback strategies
            name_candidates = [
                words[words.index("Name") + 1] if "Name" in words else "",
                next((word for word in words if len(word.split()) > 1 and word.isupper()), "")
            ]
            extracted_info["Name"] = next((name for name in name_candidates if name), "Unknown")

            # ID extraction
            id_keywords = ["Permanent Account Number Card", "Permanent Account Number"]
            id_index = next((words.index(kw) + 1 for kw in id_keywords if kw in words), -1)
            extracted_info["ID"] = words[id_index] if id_index != -1 else "Unknown"

            # Father's Name extraction
            father_keywords = ["Father's Name", "Father Name"]
            father_index = next((words.index(kw) + 1 for kw in father_keywords if kw in words), -1)
            extracted_info["Father's Name"] = words[father_index] if father_index != -1 else "Unknown"

            # Date of Birth extraction
            dob_candidates = [
                word for word in words 
                if re.match(r'\d{2}/\d{2}/\d{4}', word)
            ]
            
            if dob_candidates:
                try:
                    extracted_info["DOB"] = datetime.strptime(dob_candidates[0], "%d/%m/%Y")
                except ValueError:
                    extracted_info["DOB"] = "Unknown"
            else:
                extracted_info["DOB"] = "Unknown"

        except Exception as detail_err:
            logging.warning(f"Detailed extraction failed: {detail_err}")
            # Fallback to basic extraction if detailed fails
            extracted_info = {
                "ID": words[0] if words else "Unknown",
                "Name": words[1] if len(words) > 1 else "Unknown",
                "Father's Name": "Unknown",
                "DOB": "Unknown",
                "ID Type": "PAN"
            }

    except Exception as e:
        logging.error(f"Error in information extraction: {e}")
        extracted_info = {
            "ID": "Unknown",
            "Name": "Unknown",
            "Father's Name": "Unknown",
            "DOB": "Unknown",
            "ID Type": "PAN"
        }

    return extracted_info

def extract_information1(data_string):
    try:
        # Similar robust extraction for Aadhar
        updated_data_string = data_string.replace(".", "")
        words = [word.strip() for word in updated_data_string.split("|") if len(word.strip()) > 2]
        
        extracted_info = {
            "ID": "",
            "Name": "",
            "Gender": "",
            "DOB": "",
            "ID Type": "AADHAR"
        }

        # Robust extraction strategies
        try:
            # Name extraction
            name_candidates = [
                words[words.index("DOB") - 1] if "DOB" in words else "",
                next((word for word in words if len(word.split()) > 1 and word.isupper()), "")
            ]
            extracted_info["Name"] = next((name for name in name_candidates if name), "Unknown")

            # Gender extraction
            gender_candidates = [word for word in words if word.lower() in {"male", "female"}]
            extracted_info["Gender"] = gender_candidates[0].capitalize() if gender_candidates else "Unknown"

            # Aadhar ID extraction with multiple patterns
            id_patterns = [
                re.compile(r'^\d{4} \d{4} \d{4}$'),  # XXXX XXXX XXXX
                re.compile(r'^\d{12}$'),  # 12 digit number
                re.compile(r'^\d{4}$')    # 4 digit pattern
            ]

            for pattern in id_patterns:
                id_matches = [word for word in words if pattern.match(word)]
                if id_matches:
                    extracted_info["ID"] = " ".join(id_matches)
                    break
            
            if not extracted_info["ID"]:
                extracted_info["ID"] = "Unknown"

            # DOB extraction
            dob_candidates = [
                word for word in words 
                if re.match(r'\d{2}/\d{2}/\d{4}', word)
            ]
            
            if dob_candidates:
                try:
                    extracted_info["DOB"] = datetime.strptime(dob_candidates[0], "%d/%m/%Y")
                except ValueError:
                    extracted_info["DOB"] = "Unknown"
            else:
                extracted_info["DOB"] = "Unknown"

        except Exception as detail_err:
            logging.warning(f"Detailed Aadhar extraction failed: {detail_err}")
            # Fallback to basic extraction
            extracted_info = {
                "ID": words[0] if words else "Unknown",
                "Name": words[1] if len(words) > 1 else "Unknown",
                "Gender": "Unknown",
                "DOB": "Unknown",
                "ID Type": "AADHAR"
            }

    except Exception as e:
        logging.error(f"Error in Aadhar information extraction: {e}")
        extracted_info = {
            "ID": "Unknown",
            "Name": "Unknown",
            "Gender": "Unknown",
            "DOB": "Unknown",
            "ID Type": "AADHAR"
        }

    return extracted_info