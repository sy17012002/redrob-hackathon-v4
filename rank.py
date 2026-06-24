import os
import json
import argparse
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
import faiss

# FORCE STRICT CPU COMPLIANCE & OPTIMIZE FOR CORES
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(2) 
print("🔒 Execution Tier: STRICT CPU ONLY (Enforced Compliance)")

class TwoStageCPUFunnel:
    def __init__(self, current_date_str="2026-06-22"):
        self.current_date = datetime.strptime(current_date_str, "%Y-%m-%d")
        
        self.blacklisted_professions = [
            "civil", "mechanical", "electrical", "chemical", "accountant", "accounting",
            "support", "operations", "hr", "human resources", "recruiter", "recruitment",
            "sales", "marketing", "writer", "designer", "seo", "finance", "legal", 
            "admin", "executive", "project manager", "product manager", "business analyst", 
            "qa", "quality assurance", "testing", "tester", "customer", "helpdesk",
            "office manager", "branch manager", "store", "cashier", "logistics"
        ]

    def process_record(self, cand):
        profile = cand.get("profile", {})
        title = (profile.get("current_title") or "").strip().lower()
        
        if any(bad_word in title for bad_word in self.blacklisted_professions):
            if not any(ai_term in title for ai_term in ["ai", "ml", "machine learning", "data scientist"]):
                return None
            
        headline = (profile.get("headline") or "").strip().lower()
        summary = (profile.get("summary") or "").strip().lower()
        combined_text = f"{title} {headline} {summary}"

        lexical_score = 0
        ai_signals = ["ai", "ml", "llm", "nlp", "rag", "machine learning", "pytorch", "transformer", "embeddings", "vector"]
        for kw in ai_signals:
            if kw in combined_text:
                lexical_score += 10
                
        eng_signals = ["python", "engineer", "developer", "backend", "software", "data", "coding"]
        for kw in eng_signals:
            if kw in combined_text:
                lexical_score += 2

        if lexical_score == 0:
            return None

        cid = cand.get("candidate_id")
        history = cand.get("career_history", [])
        skills = cand.get("skills", [])
        signals = cand.get("redrob_signals", {})
        
        yoe = float(profile.get("years_of_experience", 0))
        location = (profile.get("location") or "").strip()
        country = (profile.get("country") or "").strip()
        loc_str = f"{location}, {country}".lower()

        narrative_chunks = [
            f"Title: {profile.get('current_title', '')}.",
            f"Headline: {profile.get('headline', '')}.",
            f"Summary: {profile.get('summary', '')}."
        ]
        for job in history[:2]:
            narrative_chunks.append(f"Job: {job.get('title')} at {job.get('company')}: {job.get('description', '')}.")
        valid_skills = [s.get("name") for s in skills[:6]]
        narrative_chunks.append("Skills: " + ", ".join(valid_skills))
        semantic_text = " ".join(narrative_chunks)

        months_dormant = 12.0
        last_active_str = signals.get("last_active_date")
        if last_active_str:
            try:
                last_active_dt = datetime.strptime(last_active_str, "%Y-%m-%d")
                months_dormant = max(0.0, (self.current_date - last_active_dt).days / 30.44)
            except ValueError:
                pass

        metadata = {
            "title": title,
            "yoe": yoe,
            "location_string": loc_str,
            "recruiter_response_rate": float(signals.get("recruiter_response_rate", 0.0)),
            "months_dormant": float(months_dormant),
            "github_activity_score": float(signals.get("github_activity_score", -1))
        }

        exp_w = ScoringEngine.get_experience_weight(yoe)
        loc_w = ScoringEngine.get_location_weight(loc_str)
        rough_score = lexical_score * exp_w * loc_w

        return {
            "candidate_id": cid,
            "semantic_text": semantic_text,
            "metadata": metadata,
            "rough_score": rough_score
        }

class ScoringEngine:
    @staticmethod
    def get_experience_weight(yoe):
        if 5.0 <= yoe <= 9.0: return 1.0
        elif yoe < 5.0: return max(0.70, yoe / 5.0)
        else: return max(0.80, 1.0 - (yoe - 9.0) * 0.04)

    @staticmethod
    def get_location_weight(loc_str):
        if "pune" in loc_str or "noida" in loc_str: return 1.0
        for city in ["bangalore", "bengaluru", "mumbai", "delhi", "hyderabad", "chennai", "kolkata"]:
            if city in loc_str: return 0.95
        return 0.80

    @staticmethod
    def get_behavioral_multiplier(metadata):
        recency = max(0.0, 1.0 - (metadata["months_dormant"] / 12.0)) if metadata["months_dormant"] < 12.0 else 0.0
        response_rate = metadata["recruiter_response_rate"]
        git_score = metadata["github_activity_score"]
        git_factor = (git_score / 100.0) if git_score >= 0 else 0.50
        
        raw_behavioral = (0.4 * response_rate) + (0.3 * recency) + (0.3 * git_factor)
        return 0.85 + (0.15 * raw_behavioral)

def main():
    # SET UP COMMAND LINE PARSING AS PER THE SUBMISSION SPECIFICATION
    parser_args = argparse.ArgumentParser(description="Redrob Hackathon v4 Ranking Pipeline")
    parser_args.add_argument("--candidates", required=True, help="Path to the input candidates.jsonl file")
    parser_args.add_argument("--out", required=True, help="Path to write the final submission CSV file")
    args = parser_args.parse_args()

    data_file = args.candidates
    output_csv = args.out

    if not os.path.exists(data_file):
        print(f"❌ Input Path Error: Cannot find '{data_file}'")
        return

    parser = TwoStageCPUFunnel()
    candidates_pool = []
    corrupted_lines = 0
    
    print("📥 Stage 1: Streaming entries through rapid lexical filter...")
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    parsed = parser.process_record(json.loads(line))
                    if parsed:
                        candidates_pool.append(parsed)
                except json.JSONDecodeError:
                    corrupted_lines += 1
                    continue
                    
    print(f"📊 Identified {len(candidates_pool)} tech profiles.")

    print("🎛️ Funneling down to the top 400 elite candidates for CPU neural ranking...")
    candidates_pool.sort(key=lambda x: -x["rough_score"])
    elite_targets = candidates_pool[:400]

    print("🚀 Stage 2: Initializing BGE Transformer model strictly on CPU...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")

    texts = [c["semantic_text"] for c in elite_targets]
    print(f"🧠 Generating vector transformations for the top {len(elite_targets)} targets...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    
    dimension = model.get_embedding_dimension()
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    jd_query = (
        "Represent this sentence for searching relevant passages: "
        "Looking for a Senior AI Engineer, founding team member. Proficient in modern machine learning systems, "
        "large language models, LLMs, fine-tuning, dense vector embeddings, information retrieval, ranking architectures, "
        "RAG systems, recommendation search pipelines, and building production infrastructure rapidly."
    )
    query_vector = model.encode([jd_query], convert_to_numpy=True)
    
    print("⚡ Searching vector space using FAISS index...")
    distances, indices = index.search(query_vector, len(elite_targets))
    
    semantic_scores = {}
    for rank_idx, cand_idx in enumerate(indices[0]):
        if cand_idx == -1: continue
        dist = float(distances[0][rank_idx])
        semantic_scores[elite_targets[cand_idx]["candidate_id"]] = (1.0 / (1.0 + dist)) ** 2

    print("📊 Evaluating multi-factor heuristic multipliers...")
    scored_output = []
    for cand in elite_targets:
        cid = cand["candidate_id"]
        meta = cand["metadata"]
        
        sem_score = semantic_scores.get(cid, 0.0)
        final_score = float(sem_score * ScoringEngine.get_experience_weight(meta["yoe"]) * ScoringEngine.get_location_weight(meta["location_string"]) * ScoringEngine.get_behavioral_multiplier(meta))
        
        clean_title = meta['title'].title() if meta['title'] else "AI Engineer"
        clean_location = meta['location_string'].split(',')[0].title() if meta['location_string'] else "India"
        
        reasoning = (
            f"{clean_title} based in {clean_location} with {meta['yoe']} YOE. "
            f"Platform response rate: {int(meta['recruiter_response_rate'] * 100)}%, "
            f"GitHub Activity Score: {int(meta['github_activity_score'])}."
        )
        
        scored_output.append({
            "candidate_id": cid,
            "score": round(final_score, 4),
            "reasoning": reasoning
        })

    scored_output.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    
    top_rows = scored_output[:100]
    for idx, entry in enumerate(top_rows):
        entry["rank"] = idx + 1

    submission_df = pd.DataFrame(top_rows)[["candidate_id", "rank", "score", "reasoning"]]
    submission_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"✨ Success! Compliant, high-precision CPU CSV file generated at: '{output_csv}'")

if __name__ == "__main__":
    main()