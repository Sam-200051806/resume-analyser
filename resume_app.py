from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import uuid
import shutil
import uvicorn
from typing import Optional, List
import PyPDF2
import pytesseract
from PIL import Image
from groq import Groq
from dotenv import load_dotenv
import os

# Load variables from .env
load_dotenv()

# Access the API key
groq_api_key = os.getenv("GROQ_API_KEY")

app = FastAPI(title="Resume Analyzer API")

# # Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure storage directories
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
groq_client = Groq(api_key=groq_api_key)
resume_database = {}
class ResumeInfo(BaseModel):
    resume_id: str
    filename: str
    content_type: str
    extracted_info: Optional[dict] = None

class ResumeQuery(BaseModel):
    query: str

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

def extract_text_from_image(file_path):
    """Extract text from an image using OCR."""
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""

def extract_resume_info(text):
    """Use Groq to extract structured information from resume text."""
    try:
        prompt = f"""
        Extract the following information from the resume text below:
        - Full Name
        - Email
        - Phone Number
        - LinkedIn (if available)
        - Education (degrees, institutions, years)
        - Work Experience (companies, positions, dates, descriptions)
        - Skills
        - Certifications (if any)

        Format the output as JSON.

        Resume text:
        {text}
        """
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",  
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
        )

        extracted_info = response.choices[0].message.content
        import re
        import json
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', extracted_info)
        if json_match:
            extracted_info = json_match.group(1)
        
        try:
            return json.loads(extracted_info)
        except:
            return {"raw_extraction": extracted_info}
            
    except Exception as e:
        print(f"Error using Groq for extraction: {e}")
        return {"error": str(e)}

@app.post("/upload-resume/", response_model=ResumeInfo)
async def upload_resume(file: UploadFile = File(...)):
    """Upload a resume (PDF or image) and process it."""
    content_type = file.content_type
    if not (content_type.startswith("application/pdf") or content_type.startswith("image/")):
        raise HTTPException(status_code=400, detail="Only PDF or image files are accepted")
    
    # Generate unique ID and save file
    resume_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    file_path = os.path.join(UPLOAD_DIR, f"{resume_id}{file_extension}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    text = ""
    if content_type.startswith("application/pdf"):
        text = extract_text_from_pdf(file_path)
    elif content_type.startswith("image/"):
        text = extract_text_from_image(file_path)
    extracted_info = extract_resume_info(text)
    resume_info = ResumeInfo(
        resume_id=resume_id,
        filename=file.filename,
        content_type=content_type,
        extracted_info=extracted_info
    )
    resume_database[resume_id] = resume_info
    
    return resume_info

@app.get("/resumes/", response_model=List[ResumeInfo])
async def list_resumes():
    """List all uploaded resumes."""
    return list(resume_database.values())

@app.get("/resume/{resume_id}", response_model=ResumeInfo)
async def get_resume(resume_id: str):
    """Get information about a specific resume."""
    if resume_id not in resume_database:
        raise HTTPException(status_code=404, detail="Resume not found")
    return resume_database[resume_id]

@app.post("/query-resume/{resume_id}")
async def query_resume(resume_id: str, query: ResumeQuery):
    """Query information from a specific resume using Groq."""
    if resume_id not in resume_database:
        raise HTTPException(status_code=404, detail="Resume not found")
    
    resume_info = resume_database[resume_id]
    file_extension = os.path.splitext(resume_info.filename)[1]
    file_path = os.path.join(UPLOAD_DIR, f"{resume_id}{file_extension}")
    text = ""
    if resume_info.content_type.startswith("application/pdf"):
        text = extract_text_from_pdf(file_path)
    elif resume_info.content_type.startswith("image/"):
        text = extract_text_from_image(file_path)
    
    try:
        prompt = f"""
        Based on the following resume text, answer this question: {query.query}
        
        Resume text:
        {text}
        """
        
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1000,
        )
        
        answer = response.choices[0].message.content
        return {"query": query.query, "answer": answer}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying Groq: {str(e)}")

@app.delete("/resume/{resume_id}")
async def delete_resume(resume_id: str):
    """Delete a resume."""
    if resume_id not in resume_database:
        raise HTTPException(status_code=404, detail="Resume not found")
    
    # Delete the file
    file_extension = os.path.splitext(resume_database[resume_id].filename)[1]
    file_path = os.path.join(UPLOAD_DIR, f"{resume_id}{file_extension}")
    
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error deleting file: {e}")
    del resume_database[resume_id]
    
    return {"message": "Resume deleted successfully"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)