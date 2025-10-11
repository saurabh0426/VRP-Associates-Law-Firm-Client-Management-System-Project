📄 README.md for VRP Associates Law Firm AI-Powered System
Markdown

# VRP Associates Law Firm - AI-Powered Case & Document Management System

## 🌟 Overview
This project is an advanced web application designed to streamline case management, client interaction, and document analysis for VRP Associates Law Firm. It utilizes a modern **Firebase-integrated frontend** for secure user authentication and a powerful **Python/Flask backend** to leverage **Optical Character Recognition (OCR)** and **Natural Language Processing (NLP)** for automated legal document processing.

---

## 🚀 Key Features

### Frontend (Client & Lawyer Portals)
* **Secure Authentication:** Separate registration flows for **Clients** (`newUser.html`) and **Lawyers** (`newLawyer.html`) using Firebase Authentication.
* **Dashboard Views:** Dedicated dashboards for clients and lawyers to view case summaries and system statistics.
* **Case & Document Management:** Allows clients to upload documents with **drag-and-drop** functionality (`cases.html`), rename, and delete files securely via Firebase Storage.
* **Appointments:** Management and scheduling of client-lawyer appointments (`appointments.html`).
* **Profile Management:** Functionality for users and lawyers to update their personal and professional details (`profile.html`).

### Backend (AI/NLP Service)
* **OCR Processing:** Uses **`pytesseract`** to accurately extract text from uploaded images and PDF legal documents.
* **Advanced NLP:** Applies **`spaCy`** and **`TextBlob`** to the extracted text to provide:
    * **Summarization**
    * **Sentiment Analysis**
    * **Named Entity Recognition (NER)** (e.g., identifying names, dates, organizations in legal text).
* **Chatbot Integration:** Includes a front-end interface (`chatBot.html`) for an OCR-aware chatbot, likely designed to query and retrieve information directly from the analyzed documents.

---

## 💻 Technology Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Frontend UI** | HTML5, CSS3, **Bootstrap 5** | Responsive and modern user interface |
| **Authentication, DB & Storage** | **Firebase** (Auth, Firestore, Storage) | User management, Real-time case data, Document file storage |
| **Backend API** | **Flask** (Python) | RESTful API for handling file uploads and triggering OCR/NLP |
| **OCR Engine** | **pytesseract** | Text extraction from images and PDFs |
| **NLP Libraries** | **spaCy**, **TextBlob** | Advanced text analysis and processing |
| **PDF Handling** | `pdf2image` | Converts PDF pages to images for OCR processing |

---

## ⚙️ Prerequisites

Ensure you have the following installed on your development machine:

1.  **Python 3.8+**
2.  **pip** (Python package installer)
3.  **Tesseract OCR Engine:** The Tesseract executable must be installed on your system.
    * **Windows:** Download the installer and ensure the path to `tesseract.exe` is configured in `app.py`.
    * **Linux/macOS:** Install via package manager (e.g., `sudo apt install tesseract-ocr`).
4.  **Firebase Project:** A configured Firebase project with **Authentication**, **Firestore**, and **Cloud Storage** enabled.

---

## 🚀 Installation & Setup

### 1. Backend (Python/Flask) Setup

1.  **Clone the repository:**
    ```bash
    git clone <YOUR_REPO_URL>
    cd <project-root-folder>
    ```

2.  **Create a Python Virtual Environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use: .\venv\Scripts\activate
    ```

3.  **Install Python Dependencies:**
    ```bash
    # A complete list of dependencies should be placed in a requirements.txt file.
    # If no requirements.txt exists, run:
    pip install Flask Flask-CORS pytesseract spacy textblob Pillow numpy pdf2image
    ```

4.  **Download spaCy Model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

5.  **Update Tesseract Path:**
    If using Windows, you must update the path in `app.py`:
    ```python
    # Inside app.py
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" # <--- UPDATE THIS
    ```

### 2. Frontend (HTML/JS) Configuration

You must replace the placeholder Firebase configuration with your actual project credentials in all HTML files that initialize Firebase (e.g., `index.html`, `logout.html`, `chatBot.html`, etc.).

**Configuration Snippet:**
```javascript
// Inside your HTML files, update the firebaseConfig object:
const firebaseConfig = {
    apiKey: "YOUR_API_KEY_HERE", 
    authDomain: "YOUR_AUTH_DOMAIN_HERE", 
    projectId: "YOUR_PROJECT_ID_HERE",
    // ... rest of your project keys
};
▶️ How to Run the Application
1. Start the Backend Service
Ensure your virtual environment is active and run the Flask application:

Bash

python app.py
The service will start, typically running on http://127.0.0.1:5000/.

2. Launch the Frontend
Open the index.html file directly in your web browser.

Use the Login/Registration pages to sign up as a new lawyer or client.

Navigate to cases.html to test file upload and document analysis features.

Use uploadFile.html for a dedicated, raw test of the OCR/NLP backend service.

📜 License
This project is currently licensed under [License Type, e.g., MIT].

🤝 Contribution
We welcome contributions! Please feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.
