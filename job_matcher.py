import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from resume_rag import ResumeRAG, KNOWN_SKILLS, MetadataExtractor


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _extract_skills_from_text(text: str) -> List[str]:
    text_lower = text.lower()
    return sorted({s for s in KNOWN_SKILLS if s.lower() in text_lower})


def _extract_years_requirement(text: str) -> Tuple[int, str]:
    m = re.search(r"(\d+)\+?\s*years?\s+(?:of\s+)?(\w+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1)), m.group(2)
    return 0, ""


class JobMatcher:
    def __init__(
        self,
        resumes_dir: str = "resumes",
        db_path: str = "./chroma_db",
        semantic_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ):
        self.rag = ResumeRAG(resumes_dir=resumes_dir, db_path=db_path)
        self.collection = self.rag.collection
        self.model = self.rag.model
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self.extractor = MetadataExtractor()
        self._bm25_corpus: Optional[List[List[str]]] = None
        self._bm25_index: Optional[BM25Okapi] = None
        self._all_docs: Optional[List[Dict]] = None
        self._build_bm25_index()

    def _build_bm25_index(self):
        result = self.collection.get(include=["documents", "metadatas"])
        docs = result["documents"]
        metas = result["metadatas"]
        self._all_docs = [{"text": d, "meta": m} for d, m in zip(docs, metas)]
        corpus = [_tokenize(d) for d in docs]
        self._bm25_corpus = corpus
        self._bm25_index = BM25Okapi(corpus) if corpus else None

    def _semantic_search(self, jd_embedding: List[float], n_results: int) -> List[Dict]:
        results = self.collection.query(
            query_embeddings=[jd_embedding],
            n_results=min(n_results, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = max(0.0, 1.0 - dist)
            hits.append({"text": doc, "meta": meta, "semantic_score": similarity})
        return hits

    def _bm25_search(self, query_tokens: List[str]) -> Dict[str, float]:
        if not self._bm25_index or not self._all_docs:
            return {}
        raw_scores = self._bm25_index.get_scores(query_tokens)
        max_score = raw_scores.max() if raw_scores.max() > 0 else 1.0
        normalized = raw_scores / max_score
        doc_scores: Dict[str, float] = {}
        for idx, score in enumerate(normalized):
            key = self._all_docs[idx]["meta"].get("resume_path", str(idx))
            doc_scores[key] = max(doc_scores.get(key, 0.0), float(score))
        return doc_scores

    def _fuse_and_rank(
        self,
        semantic_hits: List[Dict],
        bm25_scores: Dict[str, float],
        jd_text: str,
        top_k: int,
    ) -> List[Dict]:
        candidate_chunks: Dict[str, List[Dict]] = {}

        for hit in semantic_hits:
            path = hit["meta"].get("resume_path", "")
            bm25 = bm25_scores.get(path, 0.0)
            hybrid = self.semantic_weight * hit["semantic_score"] + self.bm25_weight * bm25
            hit["bm25_score"] = bm25
            hit["hybrid_score"] = hybrid
            name = hit["meta"].get("name", "Unknown")
            if name not in candidate_chunks:
                candidate_chunks[name] = []
            candidate_chunks[name].append(hit)

        for path, bm25 in bm25_scores.items():
            all_paths = {h["meta"].get("resume_path", "") for hits in candidate_chunks.values() for h in hits}
            if path not in all_paths:
                matching = [d for d in self._all_docs if d["meta"].get("resume_path") == path]
                for doc in matching[:3]:
                    sem_score = 0.3
                    hybrid = self.semantic_weight * sem_score + self.bm25_weight * bm25
                    name = doc["meta"].get("name", "Unknown")
                    if name not in candidate_chunks:
                        candidate_chunks[name] = []
                    candidate_chunks[name].append({
                        "text": doc["text"],
                        "meta": doc["meta"],
                        "semantic_score": sem_score,
                        "bm25_score": bm25,
                        "hybrid_score": hybrid,
                    })

        jd_skills = _extract_skills_from_text(jd_text)
        req_years, req_skill = _extract_years_requirement(jd_text)

        ranked = []
        for name, chunks in candidate_chunks.items():
            best_chunk = max(chunks, key=lambda c: c["hybrid_score"])
            top_chunks = sorted(chunks, key=lambda c: c["hybrid_score"], reverse=True)[:3]
            raw_score = best_chunk["hybrid_score"]

            stored_skills_json = best_chunk["meta"].get("skills", "[]")
            try:
                candidate_skills = json.loads(stored_skills_json)
            except Exception:
                candidate_skills = []

            matched_skills = sorted(set(jd_skills) & set(candidate_skills))
            skill_bonus = min(0.1, len(matched_skills) * 0.012)
            exp_years = best_chunk["meta"].get("experience_years", 0)
            exp_penalty = -0.05 if req_years > 0 and exp_years < req_years else 0.0
            final_score = min(1.0, raw_score + skill_bonus + exp_penalty)
            match_score = round(final_score * 100, 1)

            relevant_excerpts = [c["text"][:300].strip() for c in top_chunks]

            top_section = top_chunks[0]["meta"].get("section", "profile") if top_chunks else "profile"
            skill_str = ", ".join(matched_skills[:5]) if matched_skills else "general background"
            reasoning = (
                f"Candidate scored {match_score}/100. "
                f"Strong alignment in [{top_section}] section. "
                f"Key matching skills: {skill_str}. "
                f"Experience: {exp_years} years"
                + (f" (requirement: {req_years}+)" if req_years else "") + "."
            )

            ranked.append({
                "candidate_name": name,
                "resume_path": best_chunk["meta"].get("resume_path", ""),
                "match_score": match_score,
                "matched_skills": matched_skills,
                "relevant_excerpts": relevant_excerpts,
                "reasoning": reasoning,
            })

        ranked.sort(key=lambda x: x["match_score"], reverse=True)
        return ranked[:top_k]

    def match(self, jd_text: str, top_k: int = 10) -> Dict[str, Any]:
        self._build_bm25_index()
        start = time.time()
        jd_embedding = self.model.encode(jd_text).tolist()
        fetch_n = min(self.collection.count(), top_k * 5)
        semantic_hits = self._semantic_search(jd_embedding, n_results=fetch_n)
        bm25_scores = self._bm25_search(_tokenize(jd_text))
        top_matches = self._fuse_and_rank(semantic_hits, bm25_scores, jd_text, top_k)
        latency = round(time.time() - start, 3)

        output = {
            "job_description": jd_text[:500].strip() + ("..." if len(jd_text) > 500 else ""),
            "retrieval_latency_seconds": latency,
            "top_matches": top_matches,
        }
        return output

    def match_from_file(self, jd_path: str, top_k: int = 10) -> Dict[str, Any]:
        with open(jd_path, "r", encoding="utf-8") as f:
            jd_text = f.read()
        return self.match(jd_text, top_k=top_k)


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python job_matcher.py <path_to_jd.txt> [top_k]")
        print("\nRunning demo with all job descriptions...\n")
        jd_dir = Path("job_descriptions")
        if not jd_dir.exists():
            print("No job_descriptions directory found.")
            return

        matcher = JobMatcher()
        for jd_file in sorted(jd_dir.glob("*.txt")):
            print(f"\n{'='*60}")
            print(f"Processing: {jd_file.name}")
            print("=" * 60)
            result = matcher.match_from_file(str(jd_file), top_k=5)
            output_path = f"match_{jd_file.stem}.json"
            with open(output_path, "w") as out:
                json.dump(result, out, indent=2)
            print(f"Top 5 matches:")
            for m in result["top_matches"]:
                print(f"  {m['match_score']:5.1f} | {m['candidate_name']:<25} | Skills: {', '.join(m['matched_skills'][:4])}")
            print(f"  Saved to {output_path}")
        return

    jd_path = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    matcher = JobMatcher()
    result = matcher.match_from_file(jd_path, top_k=top_k)
    print(json.dumps(result, indent=2))
    with open("match_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nResults saved to match_results.json")


if __name__ == "__main__":
    main()

