from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import json
import os

class ResumeScreener:
    def __init__(self, resume_folder: str, model: str = "openai/gpt-3.5-turbo"):
        self.resume_folder = resume_folder
        self.model = model
        self.db = None
        self.llm = None
        self._initialize()
    
    def _initialize(self):
        """Load resumes and create vector DB"""
        documents = []
        
        # Load all PDFs
        if os.path.exists(self.resume_folder):
            for file in os.listdir(self.resume_folder):
                if file.endswith(".pdf"):
                    try:
                        loader = PyPDFLoader(os.path.join(self.resume_folder, file))
                        documents.extend(loader.load())
                    except Exception as e:
                        print(f"Error loading {file}: {e}")
        
        if not documents:
            print(f"⚠️  No PDFs found in {self.resume_folder}")
            print("Creating dummy documents for testing...")
            from langchain_core.documents import Document
            documents = [
                Document(page_content="Python developer with 5 years FastAPI experience"),
                Document(page_content="Java backend engineer, 3 years Spring Boot")
            ]
        
        # Split
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        
        # Create DB using free local embeddings (no API key needed)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.db = FAISS.from_documents(chunks, embeddings)
        print(f"✓ Loaded {len(chunks)} chunks")
    
    def score_candidate(self, job_description: str, resume_text: str):
        """Score resume against job"""
        
        # Retrieve relevant chunks
        relevant_chunks = self.db.similarity_search(job_description, k=3)
        context = "\n".join([doc.page_content for doc in relevant_chunks])
        
        # Create prompt
        prompt = PromptTemplate(
            template="""Job Description:
{job_desc}

Resume Context:
{resume_chunk}

Rate this candidate 1-10 for the role. Return ONLY valid JSON with:
{{"score": <number 1-10>, "reason": "<brief explanation>", "match_keywords": ["keyword1", "keyword2"]}}""",
            input_variables=["job_desc", "resume_chunk"]
        )
        
        # Call LLM via OpenRouter
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=0,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        chain = prompt | self.llm
        
        try:
            response = chain.invoke({
                "job_desc": job_description,
                "resume_chunk": context
            })
            
            # Parse JSON from response
            response_text = response.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            result = json.loads(response_text.strip())
        except json.JSONDecodeError:
            result = {
                "score": 5,
                "reason": "JSON parse error",
                "match_keywords": []
            }
        except Exception as e:
            result = {
                "score": 0,
                "reason": f"Error: {str(e)}",
                "match_keywords": []
            }
        
        return result
    
    def score_resume_direct(self, job_description: str, resume_text: str):
        """Score a single resume's full text directly against a job description (no retrieval needed)."""
        prompt = PromptTemplate(
            template="""Job Description:
{job_desc}

Candidate Resume:
{resume_text}

Rate this candidate 1-10 for the role. Return ONLY valid JSON with:
{{"score": <number 1-10>, "reason": "<brief explanation>", "match_keywords": ["keyword1", "keyword2"]}}""",
            input_variables=["job_desc", "resume_text"]
        )

        self.llm = ChatOpenAI(
            model=self.model,
            temperature=0,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        chain = prompt | self.llm

        try:
            response = chain.invoke({
                "job_desc": job_description,
                "resume_text": resume_text[:6000]
            })

            response_text = response.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            result = json.loads(response_text.strip())
        except json.JSONDecodeError:
            result = {"score": 5, "reason": "JSON parse error", "match_keywords": []}
        except Exception as e:
            result = {"score": 0, "reason": f"Error: {str(e)}", "match_keywords": []}

        return result