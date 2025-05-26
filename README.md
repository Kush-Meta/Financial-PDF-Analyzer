PDF Chat Analyzer with LLaMA
This project is a lightweight, local-first Streamlit application that allows users to upload financial PDFs—like 10-K or Form 4 filings—and interact with them using a conversational AI interface powered by LLaMA 3 through Ollama. The app is designed to run entirely on a personal machine using Docker and does not rely on cloud-based APIs or external services, ensuring data privacy and speed.

In addition to natural language Q&A, the app parses the PDF using pdfplumber to extract structured financial tables. Users can preview these tables inside the app and download them as CSV files for further analysis. The goal is to provide analysts, researchers, and students with a seamless way to explore dense financial documents without manually searching through pages of content.

The app includes:

LangChain-powered RAG (Retrieval-Augmented Generation) pipeline using LLaMA

Vector-based document search using FAISS

Table extraction with CSV download support

Chat history to keep track of past questions and answers

Full Docker containerization for ease of deployment and sharing

This tool is ideal for anyone exploring large volumes of unstructured or semi-structured text, particularly in financial research, compliance, or due diligence settings.

To run the app, simply build the Docker image, launch the container, and open the Streamlit interface in your browser. The architecture is modular, making it easy to extend with sentiment analysis, entity recognition, or domain-specific financial insights.
