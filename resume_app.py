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
import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import RecursiveCharacterTextSplitter
# from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Load variables from .env
load_dotenv()

# Access API keys
groq_api_key = os.getenv("GROQ_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")  # Updated for AWS region
index_name = os.getenv("INDEX_NAME", "resume-index")

# Initialize Groq client
groq_client = Groq(api_key=groq_api_key)

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Initialize Pinecone client
pinecone_client = Pinecone(
    api_key=pinecone_api_key
)

# Check if the index exists, and create it if it doesn't
if index_name not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(
        name=index_name,
        dimension=768,  # Adjust this based on your embedding model
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=pinecone_environment
        )
    )

# Access the index
index = pinecone_client.Index(index_name)

app = FastAPI(title="Resume Analyzer API")

# Add CORS middleware
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
resume_database = {}

class ResumeInfo(BaseModel):
    resume_id: str
    filename: str
    content_type: str
    extracted_info: Optional[dict] = None
    vector_storage_status: Optional[str] = None

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

def split_text_into_chunks(text):
    """Split text into smaller chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def store_in_pinecone(resume_id, text):
    """Store resume text as vectors in Pinecone."""
    try:
        # First delete any existing vectors for this resume
        index.delete(filter={"resume_id": resume_id})
        
        # Split text into chunks
        chunks = split_text_into_chunks(text)
        
        # Process documents and create embeddings
        vectors = []
        for i, chunk in enumerate(chunks):
            vector_id = f"{resume_id}-chunk-{i}"
            embedding = embeddings.embed_query(chunk)
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "resume_id": resume_id,
                    "chunk_id": i,
                    "text": chunk
                }
            })
        
        # Upsert vectors to Pinecone in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            index.upsert(vectors=batch)
            
        return True
    except Exception as e:
        print(f"Error storing vectors in Pinecone: {e}")
        return False

def extract_resume_info(text, resume_id):
    """Use Groq to extract structured information from resume text and store in Pinecone."""
    try:
        # Store in Pinecone
        store_success = store_in_pinecone(resume_id, text)
        
        # Extract structured info with Groq
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
            result = json.loads(extracted_info)
            result["vector_storage_status"] = "Success" if store_success else "Failed"
            return result
        except:
            return {
                "raw_extraction": extracted_info,
                "vector_storage_status": "Success" if store_success else "Failed"
            }
            
    except Exception as e:
        print(f"Error using Groq for extraction: {e}")
        return {"error": str(e), "vector_storage_status": "Failed"}

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
    
    # Extract info and store in Pinecone
    extracted_info = extract_resume_info(text, resume_id)
    
    resume_info = ResumeInfo(
        resume_id=resume_id,
        filename=file.filename,
        content_type=content_type,
        extracted_info=extracted_info,
        vector_storage_status=extracted_info.get("vector_storage_status", "Unknown")
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
    """Query information from a specific resume using vector similarity search."""
    if resume_id not in resume_database:
        raise HTTPException(status_code=404, detail="Resume not found")
    
    try:
        # Convert query to embedding
        query_embedding = embeddings.embed_query(query.query)
        
        # Search in Pinecone
        search_results = index.query(
            vector=query_embedding,
            filter={"resume_id": resume_id},
            top_k=3,
            include_metadata=True
        )
        
        # Extract relevant text chunks
        contexts = []
        for match in search_results["matches"]:
            if match["score"] > 0.7:  # Only include relevant matches
                contexts.append(match["metadata"]["text"])
        
        # If no good matches found, fall back to original approach
        if not contexts:
            resume_info = resume_database[resume_id]
            file_extension = os.path.splitext(resume_info.filename)[1]
            file_path = os.path.join(UPLOAD_DIR, f"{resume_id}{file_extension}")
            text = ""
            if resume_info.content_type.startswith("application/pdf"):
                text = extract_text_from_pdf(file_path)
            elif resume_info.content_type.startswith("image/"):
                text = extract_text_from_image(file_path)
                
            prompt = f"""
            Based on the following resume text, answer this question: {query.query}
            
            Resume text:
            {text}
            """
        else:
            # Join contexts and create prompt
            context_text = "\n\n".join(contexts)
            prompt = f"""
            Based on the following resume information, answer this question: {query.query}
            
            Resume information:
            {context_text}
            
            Answer the question based solely on the provided information.
            """
        
        # Get answer from Groq
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1000,
        )
        
        answer = response.choices[0].message.content
        return {"query": query.query, "answer": answer}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying resume: {str(e)}")

@app.delete("/resume/{resume_id}")
async def delete_resume(resume_id: str):
    """Delete a resume and its vectors from Pinecone."""
    if resume_id not in resume_database:
        raise HTTPException(status_code=404, detail="Resume not found")
    
    # Delete the file
    file_extension = os.path.splitext(resume_database[resume_id].filename)[1]
    file_path = os.path.join(UPLOAD_DIR, f"{resume_id}{file_extension}")
    
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            
        # Delete vectors from Pinecone
        index.delete(filter={"resume_id": resume_id})
            
    except Exception as e:
        print(f"Error deleting resume: {e}")
        
    del resume_database[resume_id]
    
    return {"message": "Resume deleted successfully"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)