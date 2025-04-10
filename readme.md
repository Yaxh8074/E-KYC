# E-KYC Project
A web application for automated Know Your Customer (KYC) verification using computer vision and machine learning techniques.

## Overview
This E-KYC application provides an interface where users can upload their ID card (Aadhar or PAN) and a photograph for identity verification. The system automatically extracts and verifies information through face matching and text recognition.

### Key Features
- **Face Verification**: Extracts faces from ID cards using Haarcascade and matches with the provided photograph
- **Optical Character Recognition**: Uses EasyOCR to extract text from verified ID cards
- **Database Integration**: Checks for duplicates before storing user data
- **Face Embeddings**: Utilizes FaceNet from DeepFace for secure storage of facial biometrics

### Technologies Used
- Computer Vision & Natural Language Processing
- Convolutional Neural Networks (CNNs)
- Long Short-Term Memory Networks (LSTMs)
- EasyOCR, DeepFace, Haarcascade

## Upcoming Improvements
1. **Live Face Detection**: Camera-based verification instead of static image uploads
2. **Enhanced Data Privacy**: Hashing of sensitive information **[COMPLETED]**

## Setup Instructions

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/Yaxh8074/E-KYC.git
    cd E-KYC
    ```

2. **Create and Activate a Virtual Environment**:
    ```sh
    python -m venv .venv 
    # Windows
    .\.venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Configure Database**:
    Create a `config.toml` file in the project root:
    ```toml
    [database]
    user = "your_username"
    password = "your_password"
    host = "localhost"
    database = "your_database_name"
    ```

5. **Run the Application**:
    ```sh
    streamlit run app.py
    ```

## Prerequisites
- Python 12.0
- MySQL server

## Security Notes
- Add `config.toml` to your `.gitignore` to prevent exposing database credentials
- The application logs events in the `logs` directory for debugging

## Troubleshooting
- **Database Connection Issues**: Verify your MySQL server is running and credentials are correct
- **Missing Packages**: Run `pip install -r requirements.txt` to ensure all dependencies are installed

