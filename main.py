from fastapi import FastAPI, File, UploadFile, Form
import pymupdf  # PDF parser
import arxiv    # Paper search
from google import genai
from google.genai import types
from fastapi.middleware.cors import CORSMiddleware
import json # <-- Added to parse the AI text into JSON

app = FastAPI(title="AI Research Gap Analysis Backend")

# 1. FIXED CORS: Explicit origins replace the wildcard to prevent fatal server crashes
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "https://research-agent-frontend-gold.vercel.app" # Your live frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client()

@app.get("/")
def home():
    return {"status": "Backend is running smoothly!"}

@app.post("/api/analyze")
async def analyze_research(
    file: UploadFile = File(None),  
    research_question: str = Form(...)  
):
    pdf_text = ""
    if file is not None:
        pdf_bytes = await file.read()
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            pdf_text += page.get_text()

    # NOTE: If arxiv library throws an error here, use: client = arxiv.Client(); search_results = client.results(search)
    search = arxiv.Search(
        query=research_question,
        max_results=3,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    arxiv_papers = []
    papers_context = ""
    for i, result in enumerate(search.results()):
        # 2. FIXED NAMING: Keys now match exactly what frontend JS looks for
        paper_info = {
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "published": str(result.published), 
            "summary": result.summary,
            "url": result.pdf_url 
        }
        arxiv_papers.append(paper_info)
        papers_context += f"\nPaper {i+1}:\nTitle: {result.title}\nAbstract: {result.summary}\n"

    # 3. FIXED PROMPT: Enforced strict JSON schema matching the JS mapping
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

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.3
            ),
        )
        # Parse the string text into an actual Python dictionary
        ai_analysis_dict = json.loads(response.text)
    except Exception as e:
        print(f"Gemini error: {e}")
        ai_analysis_dict = {}

    # 4. FIXED RESPONSE: Sent the exact variables expected by the frontend tab panels
    return {
        "status": "success",
        "research_question": research_question,
        "papers": arxiv_papers, 
        "matrix": ai_analysis_dict.get("comparison", []),
        "gaps": ai_analysis_dict.get("potential_gaps", []),
        "approach": ai_analysis_dict.get("suggested_approach", {})
    }