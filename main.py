from fastapi import FastAPI, File, UploadFile, Form
import pymupdf  # PDF parser
import arxiv    # Paper search
from google import genai
from google.genai import types
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="AI Research Gap Analysis Backend")
# Enable CORS for your frontend teammates
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all frontend origins for hackathon ease
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Initialize the Gemini client (reads from $env:GEMINI_API_KEY)
client = genai.Client()

@app.get("/")
def home():
    return {"status": "Backend is running smoothly!"}

@app.post("/api/analyze")
async def analyze_research(
    file: UploadFile = File(None),  # Optional uploaded PDF
    research_question: str = Form(...)  # Research topic or question
):
    """
    Backend Endpoint for Frontend Teammates:
    1. Extracts text from an uploaded research PDF (if provided).
    2. Fetches relevant papers from arXiv based on the research question.
    3. Passes everything to Gemini to generate the comparison matrix and research gaps.
    """
    
    # Step 1: Extract text from uploaded PDF (if any)
    pdf_text = ""
    if file is not None:
        pdf_bytes = await file.read()
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            pdf_text += page.get_text()

    # Step 2: Fetch related papers from arXiv automatically
    search = arxiv.Search(
        query=research_question,
        max_results=3,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    arxiv_papers = []
    papers_context = ""
    for i, result in enumerate(search.results()):
        paper_info = {
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "year": result.published.year,
            "summary": result.summary,
            "link": result.pdf_url
        }
        arxiv_papers.append(paper_info)
        papers_context += f"\nPaper {i+1}:\nTitle: {result.title}\nAbstract: {result.summary}\n"

    # Step 3: Construct the Prompt forcing structured JSON output
    prompt = f"""
    You are an expert AI Research Assistant. Analyze the following research question and context.
    
    Research Question: {research_question}
    
    Uploaded Paper Text Excerpt:
    {pdf_text[:3000] if pdf_text else "No specific PDF uploaded, rely on general domain knowledge and arXiv results."}
    
    Related arXiv Papers Found:
    {papers_context}
    
    Provide a detailed analysis in valid JSON format with the following keys:
    1. "comparison": A list of objects containing fields: "paper_title", "methodology", "dataset", "key_result", "limitation".
    2. "potential_gaps": A list of objects containing fields: "gap_description", "supporting_evidence", "related_limitation", "explanation".
    3. "suggested_approach": An object containing fields: "proposed_solution", "suggested_methodology", "possible_dataset", "possible_technologies".
    """

    # Step 4: Call Gemini Model
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.3
            ),
        )
        ai_analysis = response.text
    except Exception as e:
        ai_analysis = f"Error generating AI analysis: {str(e)}"

    # Step 5: Return comprehensive JSON for your frontend team
    return {
        "status": "success",
        "research_question": research_question,
        "relevant_papers": arxiv_papers,
        "ai_analysis": ai_analysis
    }