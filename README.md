# doc-to-text

A web app for extracting text from documents.

## Project Structure

```
doc-to-text/
├── frontend/   # Next.js 14 app (TypeScript + Tailwind CSS)
└── backend/    # Python FastAPI app
```

## Frontend

Built with Next.js 14, TypeScript, and Tailwind CSS.

```bash
cd frontend
npm install
npm run dev
# Runs on http://localhost:3000
```

## Backend

Built with FastAPI. Uses PyMuPDF for document parsing.

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload
# Runs on http://localhost:8000
```
