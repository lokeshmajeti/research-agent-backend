import os
import traceback
import pymupdf  # PDF parser
import arxiv    # Paper search
from google import genai
from google.genai import types
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import json
from typing import Optional

app = FastAPI(title="AI Research Gap Analysis Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "https://research-agent-frontend-gold.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "Backend is running smoothly!"}

@app.post("/api/analyze")
async def analyze_research(
    research_question: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    try:
        # 1. Verify API Key
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return {
                "status": "error",
                "matrix": [{"paper": "CONFIGURATION ERROR", "methodology": "Missing API Key", "dataset": "N/A", "key_result": "Failed", "limitation": "GEMINI_API_KEY is not set in Render Environment variables."}],
                "gaps": [{"title": "API Key Missing", "description": "Please add your GEMINI_API_KEY to the Render dashboard environment settings."}],
                "approach": {"summary": "Configure environment variables.", "methodology": "Add GEMINI_API_KEY on Render."}
            }

        client = genai.Client(api_key=api_key)
        
        # 2. Optional PDF Extraction
        pdf_text = ""
        if file is not None:
            pdf_bytes = await file.read()
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            for page in doc:
                pdf_text += page.get_text()

        # 3. ArXiv Search using new Client syntax
        search = arxiv.Search(
            query=research_question,
            max_results=3,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        arxiv_papers = []
        papers_context = ""
        arxiv_client = arxiv.Client()
        for i, result in enumerate(arxiv_client.results(search)):
            paper_info = {
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "published": str(result.published), 
                "summary": result.summary,
                "url": result.pdf_url 
            }
            arxiv_papers.append(paper_info)
            papers_context += f"\nPaper {i+1}:\nTitle: {result.title}\nAbstract: {result.summary}\n"

        # 4. Gemini Generation (Using stable gemini-1.5-flash model)
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
            model='gemini-1.5-flash',  # Updated to stable flash model string
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.3
            ),
        )
        ai_analysis_dict = json.loads(response.text)

        return {
            "status": "success",
            "research_question": research_question,
            "papers": arxiv_papers, 
            "matrix": ai_analysis_dict.get("comparison", []),
            "gaps": ai_analysis_dict.get("potential_gaps", []),
            "approach": ai_analysis_dict.get("suggested_approach", {})
        }

    except Exception as e:
        # Catch-all to display the exact error in the UI table instead of a 500 crash
        err_msg = str(e)
        print(f"Server Exception: {err_msg}")
        return {
            "status": "success",
            "research_question": research_question,
            "papers": [],
            "matrix": [{
                "paper": "RUNTIME EXCEPTION",
                "methodology": "Backend caught error",
                "dataset": "ERROR",
                "key_result": "Failed",
                "limitation": err_msg
            }],
            "gaps": [{"title": "Exception Occurred", "description": err_msg}],
            "approach": {"summary": "Check backend logs or variables.", "methodology": traceback.format_exc()}
        }