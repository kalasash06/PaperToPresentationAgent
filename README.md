# Paper-to-Presentation & Viva Agent

## About the Project

Paper-to-Presentation & Viva Agent is an AI-powered application that helps students understand research papers and prepare for presentations and viva examinations.

Users can upload a research paper in PDF format and interact with an AI assistant that generates summaries, explains important concepts, creates PowerPoint presentations and prepares viva questions with answers.

The application uses multiple specialised agents and Retrieval-Augmented Generation (RAG) to generate responses based on the uploaded research paper.

## Features

- **Research Paper Summary:** Generates a structured summary of the uploaded paper.
- **Concept Explanation:** Identifies and explains important technical concepts.
- **PowerPoint Generation:** Creates a downloadable PowerPoint presentation (.pptx).
- **Viva Preparation:** Generates possible viva questions along with answers.
- **AI Chat Assistant:** Allows users to ask questions about the uploaded paper.
- **Conversation Memory:** Maintains recent conversation context to support follow-up questions.

## Technologies Used

- Python
- Streamlit – User interface
- LangChain – Building LLM chains and agent workflows
- Groq / Gemini / Ollama – Language model providers
- ChromaDB – Vector database
- HuggingFace Sentence Transformers – Text embeddings
- PyPDF – Extracting text from PDF files
- python-pptx – Generating PowerPoint presentations

## How It Works

1. The user uploads a research paper in PDF format.
2. Text is extracted from the PDF.
3. The extracted text is divided into smaller chunks.
4. Embeddings are generated and stored in ChromaDB.
5. The user's request is classified by an intent router.
6. The appropriate agent processes the request using relevant information retrieved from the paper.
7. The application displays the generated response or provides a downloadable PowerPoint presentation.

## AI Agents

The application uses four specialised agents:

- **Summarizer Agent:** Generates a structured summary of the research paper.
- **Concept-Extractor Agent:** Identifies important concepts, methods and contributions and explains them.
- **Slide-Generator Agent:** Organises the paper's content into presentation slides and generates a PPTX file.
- **Viva Agent:** Generates potential examination questions and model answers.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/kalasash06/PaperToPresentationAgent.git
cd PaperToPresentationAgent
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

For Windows:

```powershell
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a `.env` file using `.env.example` as a reference.

Add your preferred LLM provider and its API key. For example:

```env
LLM_PROVIDER=groq
GROQ_MODEL=openai/gpt-oss-120b
GROQ_API_KEY=your_api_key
```

### 6. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## Project Structure

| File | Description |
|---|---|
| `app.py` | Main Streamlit application |
| `agents.py` | Agent workflows and LLM chains |
| `prompts.py` | Prompts used by the agents |
| `rag.py` | PDF processing, chunking and retrieval |
| `pptx_builder.py` | PowerPoint presentation generation |
| `config.py` | LLM provider configuration |
| `test_basic.py` | Basic project tests |

## Limitations

- Scanned PDFs may require OCR before processing.
- Complex PDF layouts may affect text extraction.
- Generated summaries, slides and answers should be reviewed for accuracy.

## Future Improvements

- Support for more document formats.
- Improved presentation themes and customisation.
- Better handling of complex research papers.
- Additional viva practice and evaluation features.

## Author

Developed as an academic project to explore Agentic AI, Large Language Models, Retrieval-Augmented Generation and multi-agent systems.
