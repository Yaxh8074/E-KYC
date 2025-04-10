# import os
# import logging
# import streamlit as st
# import traceback
# import pandas as pd
# from datetime import datetime
# import hashlib
# import cv2
# from PIL import Image
# import numpy as np
# import io

# # Import modules
# from preprocess import read_image, extract_id_card, save_image
# from ocr_engine import extract_text
# from postprocess import extract_information, extract_information1
# from face_verification import detect_and_extract_face, deepface_face_comparison, get_face_embeddings
# from sql_connection import insert_records, fetch_records, check_duplicacy, insert_records_aadhar, fetch_records_aadhar, check_duplicacy_aadhar

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

# def process_selfie(selfie_file):
#     """Process a selfie file from webcam to make it compatible with the verification system"""
#     try:
#         # Read the image as bytes
#         img_bytes = selfie_file.getvalue()
        
#         # Convert to a format OpenCV can work with
#         img_array = np.asarray(bytearray(img_bytes), dtype=np.uint8)
#         img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
#         # Save to disk for compatibility with existing verification code
#         face_path = os.path.join(os.getcwd(), "data/02_intermediate_data", "selfie_image.jpg")
#         cv2.imwrite(face_path, img)
        
#         logging.info(f"Saved selfie image to {face_path}")
#         return face_path, img
#     except Exception as e:
#         logging.error(f"Error processing selfie: {str(e)}")
#         return None, None

# def main_content(image_file, face_data, option, is_face_from_webcam=False):
#     try:
#         # Input validation
#         if not image_file:
#             st.warning("Please upload an ID card image.")
#             return
            
#         if not face_data and not is_face_from_webcam:
#             st.warning("Please upload a face image or take a selfie.")
#             return

#         # Read ID image
#         image = read_image(image_file, is_uploaded=True)
        
#         if image is None:
#             st.error("Unable to read ID card image. Please try again.")
#             return
            
#         # Extract ID card region of interest
#         image_roi, _ = extract_id_card(image)
        
#         # Process face image based on source
#         if is_face_from_webcam:
#             # Using the path that was already saved by process_selfie
#             face_image_path1 = face_data
            
#             # For debugging
#             st.write(f"Using webcam image from: {face_image_path1}")
#             # Show the selfie for debug purposes
#             selfie_img = cv2.imread(face_image_path1)
#             if selfie_img is not None:
#                 selfie_img_rgb = cv2.cvtColor(selfie_img, cv2.COLOR_BGR2RGB)
#                 st.image(selfie_img_rgb, caption="Processed Selfie", width=200)
#             else:
#                 st.error("Could not read saved selfie image")
#         else:
#             # Face image is an uploaded file
#             face_image = read_image(face_data, is_uploaded=True)
#             if face_image is None:
#                 st.error("Unable to read face image. Please try again.")
#                 return
                
#             face_image_path1 = save_image(face_image, "face_image.jpg", path="data/02_intermediate_data")
        
#         # Face extraction from ID card
#         face_image_path2 = detect_and_extract_face(img=image_roi)
        
#         if face_image_path2 is None:
#             st.error("No face detected in ID card. Please upload a clearer image.")
#             return
            
#         # For debugging - show the extracted face from ID
#         id_face_img = cv2.imread(face_image_path2)
#         if id_face_img is not None:
#             id_face_img_rgb = cv2.cvtColor(id_face_img, cv2.COLOR_BGR2RGB)
#             st.image(id_face_img_rgb, caption="Face Extracted from ID", width=200)
        
#         # Face verification
#         is_face_verified = deepface_face_comparison(
#             image1_path=face_image_path1, 
#             image2_path=face_image_path2
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
#         st.subheader("Verification Result")
#         if is_face_verified:
#             st.success("Face Verification: VERIFIED ✅")
#         else:
#             st.error("Face Verification: NOT VERIFIED ❌")
#             st.warning("The face in the selfie/uploaded image does not match the face on the ID card.")
#             return  # Stop processing if verification fails

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
#         text_info['Embedding'] = get_face_embeddings(face_image_path1)

#         # Insert records
#         if option == "PAN":
#             insert_records(text_info)
#         else:
#             insert_records_aadhar(text_info)

#         st.success("Registration Successful!")

#     except Exception as e:
#         st.error(f"An error occurred: {str(e)}")
#         logging.error(f"Error in main_content: {traceback.format_exc()}")
#         st.exception(e)  # Show detailed error for debugging
        
# # Main Streamlit app
# def main():
#     setup_logging()
#     set_page_style()

#     st.title("E-KYC Registration using OCR and Computer Vision")
#     st.write("Developed by 22MID0081 and 22MID0108")
    
#     # Initialize session state for selfie
#     if 'selfie_taken' not in st.session_state:
#         st.session_state.selfie_taken = False
#     if 'selfie_path' not in st.session_state:
#         st.session_state.selfie_path = None
    
#     # Sidebar for ID type selection
#     option = st.sidebar.radio(
#         "Select ID Card Type", 
#         ["PAN", "AADHAR"], 
#         index=0
#     )

#     # ID Card image uploader
#     image_file = st.file_uploader(
#         f"Upload {option} Card Image", 
#         type=['png', 'jpg', 'jpeg']
#     )
    
#     # Face verification options
#     face_option = st.radio(
#         "Choose face verification method:",
#         ["Upload Image", "Take Selfie"]
#     )
    
#     uploaded_face_image = None
    
#     if face_option == "Upload Image":
#         uploaded_face_image = st.file_uploader(
#             "Upload Face Image", 
#             type=['png', 'jpg', 'jpeg']
#         )
#         is_face_from_webcam = False
#         face_data = uploaded_face_image
#     else:  # Take Selfie option
#         is_face_from_webcam = True
#         st.write("Click below to capture your selfie")
#         selfie = st.camera_input("Take a selfie")
        
#         if selfie is not None and not st.session_state.selfie_taken:
#             # Process and save the selfie image
#             selfie_path, _ = process_selfie(selfie)
#             if selfie_path:
#                 st.session_state.selfie_path = selfie_path
#                 st.session_state.selfie_taken = True
#                 st.success("Selfie captured successfully!")
        
#         face_data = st.session_state.selfie_path if st.session_state.selfie_taken else None
        
#         # Button to retake selfie
#         if st.session_state.selfie_taken and st.button("Retake Selfie"):
#             st.session_state.selfie_taken = False
#             st.session_state.selfie_path = None
#             st.experimental_rerun()

#     # Process registration
#     if st.button("Register"):
#         with st.spinner("Processing... Please wait"):
#             main_content(image_file, face_data, option, is_face_from_webcam)

# if __name__ == "__main__":
#     main()



#with animation:

import os
import logging
import streamlit as st
import traceback
import pandas as pd
from datetime import datetime
import hashlib
import cv2
from PIL import Image
import numpy as np
import io
import time

# Import modules
from preprocess import read_image, extract_id_card, save_image
from ocr_engine import extract_text
from postprocess import extract_information, extract_information1
from face_verification import detect_and_extract_face, deepface_face_comparison, get_face_embeddings
from sql_connection import insert_records, fetch_records, check_duplicacy, insert_records_aadhar, fetch_records_aadhar, check_duplicacy_aadhar

# Set up logging
def setup_logging():
    logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, "ekyc_logs.log"), 
        level=logging.INFO, 
        format=logging_str, 
        filemode="a"
    )

# Enhanced page styling with animation CSS
def set_page_style():
    st.set_page_config(
        page_title="E-KYC Registration using OCR and Computer Vision",
        page_icon=":id:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .reportview-container {
        background: linear-gradient(to right, #f0f2f6, #e6e9f0);
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    
    /* Face scanning animation styles */
    @keyframes scanning {
        0% { box-shadow: 0 0 0 0 rgba(0, 255, 0, 0.5); }
        50% { box-shadow: 0 0 0 10px rgba(0, 255, 0, 0.2); }
        100% { box-shadow: 0 0 0 0 rgba(0, 255, 0, 0.5); }
    }
    
    .scanning-face {
        position: relative;
        animation: scanning 2s infinite;
        border: 2px solid #00ff00;
        border-radius: 5px;
    }
    
    .scan-line {
        position: absolute;
        height: 2px;
        background-color: #00ff00;
        left: 0;
        right: 0;
        opacity: 0.7;
        animation: scan-move 2s infinite;
    }
    
    @keyframes scan-move {
        0% { top: 0; }
        50% { top: 100%; }
        100% { top: 0; }
    }
    
    .face-container {
        position: relative;
        display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)

# Hash sensitive information
def hash_id(id_value):
    try:
        hash_object = hashlib.sha256(str(id_value).encode())
        return hash_object.hexdigest()
    except Exception as e:
        logging.error(f"Error hashing ID: {e}")
        return id_value

def add_scan_effect_to_image(img):
    """Add scanning animation overlay to the image for display"""
    # Convert to RGB if needed
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img
        
    # Create PIL Image for easy manipulation
    pil_img = Image.fromarray(img_rgb)
    
    # Return image with HTML wrapper that adds animation
    return pil_img

def process_selfie(selfie_file):
    """Process a selfie file from webcam to make it compatible with the verification system"""
    try:
        # Read the image as bytes
        img_bytes = selfie_file.getvalue()
        
        # Convert to a format OpenCV can work with
        img_array = np.asarray(bytearray(img_bytes), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # Save to disk for compatibility with existing verification code
        face_path = os.path.join(os.getcwd(), "data/02_intermediate_data", "selfie_image.jpg")
        cv2.imwrite(face_path, img)
        
        logging.info(f"Saved selfie image to {face_path}")
        return face_path, img
    except Exception as e:
        logging.error(f"Error processing selfie: {str(e)}")
        return None, None

def display_face_with_animation(image_path, caption):
    """Display face image with scanning animation"""
    img = cv2.imread(image_path)
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert image to bytes for HTML display
        is_success, buffer = cv2.imencode(".png", img_rgb)
        io_buf = io.BytesIO(buffer)
        
        # Display with scanning animation
        st.markdown(f"""
        <div class="face-container">
            <img src="data:image/png;base64,{base64.b64encode(io_buf.getvalue()).decode()}" 
                 width="200" class="scanning-face">
            <div class="scan-line"></div>
        </div>
        <p style="text-align: center;">{caption}</p>
        """, unsafe_allow_html=True)
    else:
        st.error(f"Could not read image from {image_path}")

def simulate_face_scan(image_placeholder, image_path, success_message="Face Detected"):
    """Create a simulated scanning animation sequence"""
    img = cv2.imread(image_path)
    if img is None:
        image_placeholder.error("Could not read image")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Show sequential scanning
    for i in range(5):
        # Create a copy of the image and add a horizontal line at different positions
        scan_img = img_rgb.copy()
        height, width = scan_img.shape[:2]
        line_pos = int((i / 4) * height)
        
        # Draw a green line at the current position
        cv2.line(scan_img, (0, line_pos), (width, line_pos), (0, 255, 0), 2)
        
        # Add a semi-transparent green overlay for scanned area
        overlay = img_rgb.copy()
        cv2.rectangle(overlay, (0, 0), (width, line_pos), (0, 255, 0), -1)
        alpha = 0.2  # Transparency factor
        scan_img = cv2.addWeighted(overlay, alpha, scan_img, 1 - alpha, 0)
        
        # Add a rectangle around the face area
        cv2.rectangle(scan_img, (10, 10), (width-10, height-10), (0, 255, 0), 2)
        
        # Display the animation frame
        image_placeholder.image(scan_img, caption=f"Scanning... {i+1}/5", width=200)
        time.sleep(0.3)
    
    # Show final result
    image_placeholder.image(img_rgb, caption=success_message, width=200)
    
    # Add a success indicator
    st.success("✅ " + success_message)

def main_content(image_file, face_data, option, is_face_from_webcam=False):
    try:
        # Input validation
        if not image_file:
            st.warning("Please upload an ID card image.")
            return
            
        if not face_data and not is_face_from_webcam:
            st.warning("Please upload a face image or take a selfie.")
            return

        # Create placeholders for animations
        selfie_placeholder = st.empty()
        id_face_placeholder = st.empty()
        verification_placeholder = st.empty()
        
        with st.spinner("Processing ID card..."):
            # Read ID image
            image = read_image(image_file, is_uploaded=True)
            
            if image is None:
                st.error("Unable to read ID card image. Please try again.")
                return
                
            # Extract ID card region of interest
            image_roi, _ = extract_id_card(image)
        
        # Process face image based on source
        if is_face_from_webcam:
            # Using the path that was already saved by process_selfie
            face_image_path1 = face_data
            
            # Show the animation for selfie processing
            st.markdown("### Processing Selfie")
            simulate_face_scan(selfie_placeholder, face_image_path1, "Selfie Processed")
        else:
            # Face image is an uploaded file
            face_image = read_image(face_data, is_uploaded=True)
            if face_image is None:
                st.error("Unable to read face image. Please try again.")
                return
                
            face_image_path1 = save_image(face_image, "face_image.jpg", path="data/02_intermediate_data")
            
            # Show the animation for uploaded face processing
            st.markdown("### Processing Uploaded Face")
            simulate_face_scan(selfie_placeholder, face_image_path1, "Face Image Processed")
        
        with st.spinner("Extracting face from ID card..."):
            # Face extraction from ID card
            face_image_path2 = detect_and_extract_face(img=image_roi)
            
            if face_image_path2 is None:
                st.error("No face detected in ID card. Please upload a clearer image.")
                return
        
        # Show the animation for ID card face extraction
        st.markdown("### Extracting Face from ID Card")
        simulate_face_scan(id_face_placeholder, face_image_path2, "Face Extracted from ID")
        
        # Face verification with animation
        st.markdown("### Performing Face Verification")
        verification_progress = st.progress(0)
        verification_status = st.empty()
        
        for i in range(101):
            verification_progress.progress(i)
            if i < 50:
                verification_status.info(f"Analyzing facial features... {i*2}%")
            else:
                verification_status.info(f"Comparing faces... {i*2-100}%")
            time.sleep(0.02)
        
        # Actual face verification
        is_face_verified = deepface_face_comparison(
            image1_path=face_image_path1, 
            image2_path=face_image_path2
        )
        
        verification_status.empty()
        verification_progress.empty()

        # Text extraction
        with st.spinner("Extracting information from ID card..."):
            extracted_text = extract_text(image_roi)
            
            # Extract information based on ID type
            text_info = (
                extract_information(extracted_text) if option == "PAN" 
                else extract_information1(extracted_text)
            )

        # Display extracted information
        st.subheader("Extracted Information")
        st.json(text_info)  # Display raw extracted info
        
        # Display verification status
        st.subheader("Verification Result")
        if is_face_verified:
            st.success("Face Verification: VERIFIED ✅")
            
            # Display a side-by-side comparison with a "match" indicator
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(cv2.imread(face_image_path1), cv2.COLOR_BGR2RGB), 
                        caption="Your Face", width=200)
            with col2:
                st.image(cv2.cvtColor(cv2.imread(face_image_path2), cv2.COLOR_BGR2RGB), 
                        caption="ID Card Face", width=200)
                
            st.markdown("""
            <div style="text-align: center; margin-top: -20px;">
                <div style="background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px; display: inline-block;">
                    <i class="fas fa-check-circle"></i> MATCH CONFIRMED
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Face Verification: NOT VERIFIED ❌")
            st.warning("The face in the selfie/uploaded image does not match the face on the ID card.")
            return  # Stop processing if verification fails

        # Hash ID
        original_id = text_info['ID']
        text_info['ID'] = hash_id(text_info['ID'])

        # Duplicate check
        is_duplicate = (
            check_duplicacy(text_info) if option == "PAN" 
            else check_duplicacy_aadhar(text_info)
        )

        if is_duplicate:
            st.warning(f"User already registered with ID {original_id}")
            return

        # Process and insert record
        try:
            text_info['DOB'] = (
                text_info['DOB'].strftime('%Y-%m-%d') 
                if isinstance(text_info['DOB'], datetime) 
                else str(text_info['DOB'])
            )
        except Exception as date_err:
            logging.warning(f"Date formatting error: {date_err}")

        # Get face embeddings with animation
        with st.spinner("Generating secure face signature..."):
            # Add a progress bar for embedding generation
            embedding_progress = st.progress(0)
            for i in range(101):
                embedding_progress.progress(i)
                time.sleep(0.01)
            
            text_info['Embedding'] = get_face_embeddings(face_image_path1)
            embedding_progress.empty()

        # Insert records with animation
        with st.spinner("Finalizing registration..."):
            success_placeholder = st.empty()
            
            # Show progress bar
            register_progress = st.progress(0)
            for i in range(101):
                register_progress.progress(i)
                time.sleep(0.02)
            
            # Actually insert records
            if option == "PAN":
                insert_records(text_info)
            else:
                insert_records_aadhar(text_info)
            
            register_progress.empty()
        
        # # Success animation
        # st.balloons()
        # success_placeholder.markdown("""
        # <div style="text-align: center; padding: 20px; background-color: #4CAF50; color: white; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        #     <h2>✅ Registration Successful! ✅</h2>
        #     <p>Your identity has been verified and securely stored.</p>
        # </div>
        # """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"Error in main_content: {traceback.format_exc()}")
        st.exception(e)  # Show detailed error for debugging
        
# Main Streamlit app
def main():
    setup_logging()
    set_page_style()

    st.title("E-KYC Registration using OCR and Computer Vision")
    st.write("Developed by 22MID0081 and 22MID0108")
    
    # Initialize session state for selfie
    if 'selfie_taken' not in st.session_state:
        st.session_state.selfie_taken = False
    if 'selfie_path' not in st.session_state:
        st.session_state.selfie_path = None
    
    # Import necessary for base64 encoding
    import base64
    
    # Sidebar for ID type selection
    option = st.sidebar.radio(
        "Select ID Card Type", 
        ["PAN", "AADHAR"], 
        index=0
    )

    # ID Card image uploader
    image_file = st.file_uploader(
        f"Upload {option} Card Image", 
        type=['png', 'jpg', 'jpeg']
    )
    
    # Face verification options
    face_option = st.radio(
        "Choose face verification method:",
        ["Upload Image", "Take Selfie"]
    )
    
    uploaded_face_image = None
    
    if face_option == "Upload Image":
        uploaded_face_image = st.file_uploader(
            "Upload Face Image", 
            type=['png', 'jpg', 'jpeg']
        )
        is_face_from_webcam = False
        face_data = uploaded_face_image
    else:  # Take Selfie option
        is_face_from_webcam = True
        
        # # Add the camera icon and instructions with some styling
        # st.markdown("""
        # <div style="background-color: #f0f0f0; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        #     <h4 style="margin-top: 0;">📸 Take a Selfie</h4>
        #     <p>Please position your face clearly in the frame and ensure good lighting.</p>
        # </div>
        # """, unsafe_allow_html=True)
        
        selfie = st.camera_input("Capture")
        
        if selfie is not None and not st.session_state.selfie_taken:
            # Process and save the selfie image
            selfie_path, _ = process_selfie(selfie)
            if selfie_path:
                st.session_state.selfie_path = selfie_path
                st.session_state.selfie_taken = True
                st.success("Selfie captured successfully!")
        
        face_data = st.session_state.selfie_path if st.session_state.selfie_taken else None
        
        # Button to retake selfie
        if st.session_state.selfie_taken and st.button("Retake Selfie"):
            st.session_state.selfie_taken = False
            st.session_state.selfie_path = None
            st.experimental_rerun()

    # Process registration
    if st.button("Register"):
        with st.spinner("Processing... Please wait"):
            main_content(image_file, face_data, option, is_face_from_webcam)

if __name__ == "__main__":
    main()