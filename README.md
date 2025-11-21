# LLM Q&A Project

## Overview
This project consists of a Python CLI and a Web GUI for a Question-and-Answering system powered by the Google Gemini API.

## Setup

1.  **Prerequisites**:
    *   Python 3.8+
    *   A Google Gemini API Key (Get it from [Google AI Studio](https://aistudio.google.com/app/apikey))

2.  **Installation**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configuration**:
    *   Create a `.env` file in this directory.
    *   Add your API key: `GEMINI_API_KEY=your_api_key_here`

## Usage

### CLI Application
Run the command-line interface:
```bash
python LLM_QA_CLI.py
```

### Web Application
Run the web server:
```bash
python app.py
```
Open your browser and go to `http://127.0.0.1:5000`.

## Project Structure
*   `LLM_QA_CLI.py`: CLI implementation.
*   `app.py`: Flask web server.
*   `templates/index.html`: Frontend HTML.
*   `static/style.css`: Frontend CSS.
