import os
import re
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

KNOWN_SKILLS = [
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "C", "Go", "Rust", "Ruby",
    "PHP", "Swift", "Kotlin", "Scala", "R", "MATLAB",
    "React", "Angular", "Vue.js", "Vue", "Node.js", "Node", "Django", "Flask", "FastAPI",
    "Spring", "Spring Boot", "Express",
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision", "Reinforcement Learning",
    "TensorFlow", "PyTorch", "Keras", "scikit-learn", "Scikit-learn", "JAX",
    "HuggingFace", "BERT", "GPT", "LLaMA", "LangChain", "spaCy", "NLTK",
    "AWS", "Azure", "GCP", "Google Cloud",
    "Docker", "Kubernetes", "Terraform", "Ansible", "CI/CD", "Jenkins", "GitHub Actions",
    "DevOps", "SRE", "MLOps",
    "SQL", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "Cassandra",
    "Snowflake", "BigQuery", "Redshift",
    "Apache Spark", "Spark", "Kafka", "Airflow", "Databricks", "dbt",
    "Data Science", "Data Analysis", "Data Engineering", "Data Pipeline",
    "Pandas", "NumPy", "Matplotlib", "Seaborn", "Tableau", "Power BI",
    "MLflow", "Kubeflow", "FAISS", "Pinecone", "ChromaDB", "Weaviate",
    "OpenCV", "YOLO",
    "Linux", "Unix", "Bash", "Shell",
    "Prometheus", "Grafana", "Kibana",
    "REST", "REST APIs", "GraphQL", "gRPC", "Microservices",
    "Agile", "Scrum", "Kanban", "JIRA", "Confluence",
    "Git", "GitHub", "GitLab",
    "Blockchain", "Solidity", "Ethereum", "Web3",
    "Penetration Testing", "SIEM", "CISSP", "Security",
    "Figma", "Sketch", "UX",
    "QuantLib", "Quant",
    "FreeRTOS", "ARM", "Embedded",
    "Unity", "C# (Unity)", "OpenGL",
    "Oracle", "DBA",
    "Selenium", "pytest", "TestNG", "QA",
    "Power BI", "DAX", "Excel",
    "SAFe", "PMP", "Leadership",
    "Firebase", "Jetpack Compose",
    "SwiftUI", "Xcode", "Core Data",
    "netmiko", "Cisco", "BGP", "OSPF",
    "Delta Lake", "Spark", "Databricks",
    "OpenAI", "GPT", "LLM",
]

SECTION_PATTERNS = re.compile(
    r"^(SUMMARY|OBJECTIVE|PROFESSIONAL SUMMARY|EXPERIENCE|WORK EXPERIENCE|"
    r"PROFESSIONAL EXPERIENCE|EMPLOYMENT|EDUCATION|ACADEMIC BACKGROUND|"
    r"SKILLS|TECHNICAL SKILLS|CORE COMPETENCIES|PROJECTS|NOTABLE PROJECTS|"
    r"CERTIFICATIONS|ACHIEVEMENTS|AWARDS|PUBLICATIONS|CONTACT)\s*$",
    re.MULTILINE | re.IGNORECASE,
)


class ResumeProcessor:
    def __init__(self, resumes_dir: str = "resumes"):
        self.resumes_dir = Path(resumes_dir)

    def load_resume(self, file_path: Path) -> str:
        file_path = Path(file_path)
        if file_path.suffix.lower() == ".pdf":
            return self._load_pdf(file_path)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _load_pdf(self, file_path: Path) -> str:
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(file_path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return ""

    def load_all_resumes(self) -> List[Dict[str, Any]]:
        resumes = []
        patterns = ["*.txt", "*.pdf"]
        for pattern in patterns:
            for file_path in sorted(self.resumes_dir.glob(pattern)):
                content = self.load_resume(file_path)
                if content.strip():
                    resumes.append({"path": str(file_path), "content": content, "filename": file_path.name})
        return resumes

    def chunk_by_section(self, content: str, resume_path: str) -> List[Dict[str, Any]]:
        chunks = []
        matches = list(SECTION_PATTERNS.finditer(content))

        if not matches:
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            for i, para in enumerate(paragraphs):
                chunks.append({"text": para, "section": "general", "chunk_id": i, "resume_path": resume_path})
            return chunks

        if matches[0].start() > 0:
            header = content[: matches[0].start()].strip()
            if header:
                chunks.append({"text": header, "section": "header", "chunk_id": 0, "resume_path": resume_path})

        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
            section_name = match.group(0).strip().lower()
            section_text = content[start:end].strip()

            if len(section_text) > 900:
                for sub_idx, sub_chunk in enumerate(self._split_long_section(section_text)):
                    chunks.append({
                        "text": sub_chunk,
                        "section": section_name,
                        "chunk_id": len(chunks),
                        "resume_path": resume_path,
                    })
            else:
                chunks.append({
                    "text": section_text,
                    "section": section_name,
                    "chunk_id": len(chunks),
                    "resume_path": resume_path,
                })
        return chunks

    def _split_long_section(self, text: str, max_len: int = 800) -> List[str]:
        if len(text) <= max_len:
            return [text]
        lines, current, length, result = text.split("\n"), [], 0, []
        for line in lines:
            if length + len(line) > max_len and current:
                result.append("\n".join(current))
                current, length = [line], len(line)
            else:
                current.append(line)
                length += len(line)
        if current:
            result.append("\n".join(current))
        return result


class MetadataExtractor:
    def extract_name(self, content: str) -> str:
        for line in content.strip().split("\n")[:6]:
            line = line.strip()
            if (2 <= len(line.split()) <= 4
                    and not any(ch in line for ch in ["@", ":", "|", "/", ".", "+"])
                    and re.match(r"^[A-Z]", line)):
                return line
        return content.strip().split("\n")[0].strip()

    def extract_skills(self, content: str) -> List[str]:
        content_lower = content.lower()
        return sorted({s for s in KNOWN_SKILLS if s.lower() in content_lower})

    def extract_experience_years(self, content: str) -> int:
        for pattern in [
            r"(\d+)\+?\s*years?\s+of\s+experience",
            r"(\d+)\+?\s*years?\s+experience",
            r"experience[:\s]+(\d+)\+?\s*years?",
            r"(\d+)\+?\s*yrs?\s+experience",
        ]:
            m = re.search(pattern, content, re.IGNORECASE)
            if m:
                return int(m.group(1))
        years = sorted(set(int(y) for y in re.findall(r"\b(20\d{2}|199\d)\b", content)))
        if len(years) >= 2:
            return years[-1] - years[0]
        return 0

    def extract_education(self, content: str) -> str:
        for degree in ["Ph.D", "PhD", "Doctor", "M.Tech", "M.Sc", "M.S.", "M.S", "MBA",
                       "Master", "B.Tech", "B.Sc", "B.S.", "B.S", "B.E.", "Bachelor"]:
            if re.search(re.escape(degree), content, re.IGNORECASE):
                return degree
        return "Not specified"

    def extract_all(self, content: str, file_path: str) -> Dict[str, Any]:
        return {
            "name": self.extract_name(content),
            "skills": self.extract_skills(content),
            "experience_years": self.extract_experience_years(content),
            "education": self.extract_education(content),
            "resume_path": file_path,
        }


class ResumeRAG:
    def __init__(self, resumes_dir: str = "resumes", db_path: str = "./chroma_db"):
        self.resumes_dir = resumes_dir
        self.db_path = db_path
        self.processor = ResumeProcessor(resumes_dir)
        self.extractor = MetadataExtractor()
        print("Loading embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="resumes",
            metadata={"hnsw:space": "cosine"},
        )
        print(f"Vector store ready. Current chunk count: {self.collection.count()}")

    def index_resume(self, file_path: str) -> int:
        content = self.processor.load_resume(Path(file_path))
        if not content.strip():
            return 0
        meta = self.extractor.extract_all(content, file_path)
        chunks = self.processor.chunk_by_section(content, file_path)

        ids, texts, embeddings, metadatas = [], [], [], []
        for chunk in chunks:
            chunk_id = f"{Path(file_path).stem}_{chunk['section']}_{chunk['chunk_id']}"
            ids.append(chunk_id)
            texts.append(chunk["text"])
            embeddings.append(self.model.encode(chunk["text"]).tolist())
            metadatas.append({
                "name": meta["name"],
                "skills": json.dumps(meta["skills"]),
                "experience_years": meta["experience_years"],
                "education": meta["education"],
                "resume_path": file_path,
                "section": chunk["section"],
                "filename": Path(file_path).name,
            })

        existing = set(self.collection.get(ids=ids)["ids"])
        new_ids = [i for i in range(len(ids)) if ids[i] not in existing]
        if new_ids:
            self.collection.add(
                ids=[ids[i] for i in new_ids],
                documents=[texts[i] for i in new_ids],
                embeddings=[embeddings[i] for i in new_ids],
                metadatas=[metadatas[i] for i in new_ids],
            )
        return len(new_ids)

    def index_all(self, resumes_dir: Optional[str] = None) -> Dict[str, Any]:
        directory = resumes_dir or self.resumes_dir
        resumes = self.processor.load_all_resumes() if directory == self.resumes_dir else ResumeProcessor(directory).load_all_resumes()
        total_chunks = 0
        start = time.time()
        for resume in tqdm(resumes, desc="Indexing resumes"):
            count = self.index_resume(resume["path"])
            total_chunks += count
        elapsed = round(time.time() - start, 2)
        stats = {
            "resumes_indexed": len(resumes),
            "total_chunks": total_chunks,
            "collection_size": self.collection.count(),
            "elapsed_seconds": elapsed,
        }
        print(f"\nIndexing complete: {stats}")
        return stats

    def get_collection_stats(self) -> Dict[str, Any]:
        total = self.collection.count()
        all_meta = self.collection.get(include=["metadatas"])["metadatas"]
        candidates = {m["name"] for m in all_meta}
        return {"total_chunks": total, "total_candidates": len(candidates), "candidates": sorted(candidates)}


if __name__ == "__main__":
    rag = ResumeRAG(resumes_dir="resumes", db_path="./chroma_db")
    stats = rag.index_all()
    print("\nCollection Stats:")
    print(json.dumps(rag.get_collection_stats(), indent=2))

