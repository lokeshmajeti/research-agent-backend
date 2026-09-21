import json
import os
import traceback

import arxiv
import pymupdf
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

app = FastAPI(title="AI Research Gap Analysis Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "https://research-agent-frontend-gold.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"status": "Backend is running smoothly!"}


@app.post("/api/analyze")
async def analyze_research(request: Request):
    try:
        form_data = await request.form()
        research_question = str(
            form_data.get("research_question", "AI-based crop disease detection")
        )
        file = form_data.get("file")

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return {
                "status": "error",
                "matrix": [{
                    "paper": "CONFIGURATION ERROR",
                    "methodology": "Missing API Key",
                    "dataset": "N/A",
                    "key_result": "Failed",
                    "limitation": "GEMINI_API_KEY is not set in Render.",
                }],
                "gaps": [{
                    "title": "API Key Missing",
                    "description": "Add GEMINI_API_KEY to Render settings.",
                }],
                "approach": {
                    "summary": "Configure environment variables.",
                    "methodology": "Add GEMINI_API_KEY.",
                },
            }

        client = genai.Client(api_key=api_key)

        pdf_text = ""
        if file is not None and getattr(file, "filename", None):
            pdf_bytes = await file.read()
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            try:
                for page in doc:
                    pdf_text += page.get_text()
            finally:
                doc.close()

        search = arxiv.Search(
            query=research_question,
            max_results=3,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        arxiv_papers = []
        papers_context = ""
        for index, result in enumerate(arxiv.Client().results(search)):
            paper_info = {
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "published": str(result.published),
                "summary": result.summary,
                "url": result.pdf_url,
            }
            arxiv_papers.append(paper_info)
            papers_context += (
                f"\nPaper {index + 1}:\n"
                f"Title: {result.title}\n"
                f"Abstract: {result.summary}\n"
            )

        prompt = f"""
        You are an expert AI Research Assistant. Analyze the following research question and context.

        Research Question: {research_question}
        Uploaded Paper Text Excerpt: {pdf_text[:3000] if pdf_text else "None"}
        Related arXiv Papers Found: {papers_context}

        Provide a detailed analysis in valid JSON format exactly matching these keys:
        {{
          "comparison": [
            {{"paper": "Title", "methodology": "Method", "dataset": "Data", "key_result": "Result", "limitation": "Issue"}}
          ],
          "potential_gaps": [
            {{"title": "Short Gap Name", "description": "Detailed explanation"}}
          ],
          "suggested_approach": {{"summary": "Solution direction", "methodology": "Tech Stack details"}}
        }}
        """

        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.3,
            ),
        )
        analysis = json.loads(response.text)

        return {
            "status": "success",
            "research_question": research_question,
            "papers": arxiv_papers,
            "matrix": analysis.get("comparison", []),
            "gaps": analysis.get("potential_gaps", []),
            "approach": analysis.get("suggested_approach", {}),
        }
    except Exception as error:
        error_message = str(error)
        print(f"Server Exception: {error_message}")
        print(traceback.format_exc())
        return {
            "status": "error",
            "research_question": research_question
            if "research_question" in locals()
            else "Unknown",
            "papers": [],
            "matrix": [{
                "paper": "RUNTIME EXCEPTION",
                "methodology": "Backend caught error",
                "dataset": "ERROR",
                "key_result": "Failed",
                "limitation": error_message,
            }],
            "gaps": [{
                "title": "Exception Occurred",
                "description": error_message,
            }],
            "approach": {
                "summary": "Check backend logs or variables.",
                "methodology": traceback.format_exc(),
            },
        }
