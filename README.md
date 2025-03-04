# DocuQuery: AI-Powered PDF Knowledge Assistant

## Overview
**DocuQuery** is an AI-powered web application that enables users to efficiently extract, analyze, and summarize textual content from PDF and Word documents. Leveraging **Google PaLM AI** and **Natural Language Processing (NLP)**, the system automates tasks such as **price list analysis, research paper summarization, and resume matching**. It enhances decision-making by offering insightful summaries and structured information retrieval.

## Features
- **Price List Analyzer**: Extracts item details from multiple price lists and compares prices across suppliers.
- **Research Paper Summarizer**: Generates concise summaries of research papers with different length options.
- **Resume Matcher for Hiring**: Matches resumes with job descriptions to identify the best candidates.

## Tech Stack
- **Frontend**: Streamlit (Web-based UI)
- **Backend**: Python
- **AI & NLP**: Google PaLM API, LangChain, FAISS (for vector storage)
- **Data Handling**: PyPDF2, Docx (for text extraction)
- **Environment Management**: dotenv (for API key management)

## Installation
### Prerequisites
Ensure you have **Python 3.8+** installed.

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/docuquery.git
   cd docuquery
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up your API key:
   - Create a `.env` file in the project directory.
   - Add your **Google API Key**:
     ```plaintext
     GOOGLE_API_KEY=your_api_key_here
     ```

## Usage
1. Run the application:
   ```bash
   streamlit run app.py
   ```
2. Open your browser and navigate to `http://localhost:8501`.
3. Upload your PDF/Word documents and select the desired feature from the sidebar.
4. View extracted insights, summaries, or matched resumes in real-time.
5. Download summarized content as a PDF if needed.

## Project Workflow
### Milestones
1. **Requirement Specification**: Define dependencies and setup `.env` file for API keys.
2. **Data Extraction**: Implement functions for extracting text from PDFs and Word documents.
3. **AI Model Integration**: Use Google PaLM AI for text summarization, embeddings, and resume matching.
4. **UI Development**: Build an intuitive interface using Streamlit.
5. **Testing & Optimization**: Improve accuracy, performance, and user experience.
6. **Deployment**: Deploy the application for broader accessibility.
7. **Future Enhancements**: Plan for additional features like multilingual support and deeper analysis.

## Future Enhancements
- **Support for additional document formats** beyond PDFs and Word files.
- **Cloud-based deployment** for wider accessibility.

## Contributing
Contributions are welcome! If you’d like to improve this project, please fork the repository and submit a pull request.

## Contact
For inquiries or support, reach out at `your.email@example.com`.

