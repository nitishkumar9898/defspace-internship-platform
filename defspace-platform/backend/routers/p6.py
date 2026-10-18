"""P6 — GenAI NotebookLM Skill Gap Analyzer
   TF-IDF cosine similarity for JD-resume matching.
   sklearn + NLP for skill extraction and gap analysis.
"""
import json
import re
import math
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import get_db
from models import GenAISkillAnalysis
from schemas import SkillAnalysisInput, SkillAnalysisResult, SkillEntry

router = APIRouter(prefix="/api/p6", tags=["P6 GenAI Skill Analysis"])

# Cybersecurity tech keywords for extraction
CYBER_SKILLS = [
    "python", "linux", "network security", "owasp", "siem", "splunk", "incident response",
    "vulnerability assessment", "penetration testing", "firewall", "ids/ips", "threat hunting",
    "malware analysis", "forensics", "cloud security", "aws", "azure", "docker", "kubernetes",
    "sql", "bash", "powershell", "wireshark", "nmap", "metasploit", "burp suite",
    "risk management", "iso 27001", "nist", "gdpr", "soc", "threat intelligence",
    "endpoint security", "zero trust", "devsecops", "git", "jira",
]

# Skill importance weights (industry-sourced)
SKILL_WEIGHTS = {
    "siem": 10, "splunk": 9, "incident response": 10, "python": 9, "linux": 8,
    "threat hunting": 8, "vulnerability assessment": 8, "network security": 7,
    "penetration testing": 7, "owasp": 7, "cloud security": 7, "forensics": 6,
    "malware analysis": 6, "firewall": 5, "ids/ips": 5, "aws": 5, "iso 27001": 5,
    "nist": 5, "docker": 4, "bash": 4, "powershell": 4, "wireshark": 4, "nmap": 4,
    "risk management": 6, "sql": 4, "git": 3, "jira": 2,
}

INTERVIEW_QS = [
    "Describe your process for handling a P1 security incident from detection to closure.",
    "How would you use Splunk to hunt for lateral movement in an enterprise network?",
    "Explain the OWASP Top 10 and give an example of mitigating SQL injection.",
    "What is the difference between IDS and IPS, and when would you use each?",
    "Walk me through conducting a vulnerability assessment on a web application.",
    "How does a zero-trust architecture reduce the attack surface?",
    "Explain the CIA triad and describe a real scenario where each was violated.",
    "How would you detect and respond to a ransomware infection on an endpoint?",
]

THIRTY_DAY_PLAN = [
    {"week": "Week 1", "focus": "SIEM Fundamentals", "action": "Complete Splunk Core Certified User course (free), index sample logs, build 3 dashboards.", "platform": "Splunk Training Portal"},
    {"week": "Week 2", "focus": "Incident Response", "action": "Complete TryHackMe SOC Level 1 path — 15 rooms covering triage, containment, eradication.", "platform": "TryHackMe"},
    {"week": "Week 3", "focus": "Threat Detection Scenarios", "action": "Solve 5 LetsDefend.io SOC scenarios; document playbook for each alert type.", "platform": "LetsDefend.io"},
    {"week": "Week 4", "focus": "Demonstrate & Certify", "action": "Complete CyberDefenders CTF; build GitHub portfolio with Splunk dashboards and IR reports.", "platform": "CyberDefenders.org + GitHub"},
]


def _extract_skills(text: str) -> list[str]:
    text_lower = text.lower()
    found = []
    for skill in CYBER_SKILLS:
        # word boundary match
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found.append(skill)
    return found


def _count_frequency(text: str, skill: str) -> int:
    pattern = r'\b' + re.escape(skill) + r'\b'
    return len(re.findall(pattern, text.lower()))


@router.post("/analyse", response_model=SkillAnalysisResult)
async def analyse_skills(data: SkillAnalysisInput, db: Session = Depends(get_db)):
    # TF-IDF cosine similarity between JD and resume
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform([data.jd_text, data.resume_text])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except Exception:
        sim = 0.0
    similarity_pct = round(float(sim) * 100, 2)

    # Skill extraction
    jd_skills     = _extract_skills(data.jd_text)
    resume_skills = _extract_skills(data.resume_text)
    all_jd_skills = list(set(jd_skills))

    skills_list: list[SkillEntry] = []
    missing = []

    for skill in CYBER_SKILLS:
        freq = _count_frequency(data.jd_text, skill)
        if freq == 0 and skill not in resume_skills:
            continue   # not in JD or resume → skip
        in_resume = skill in resume_skills
        weight = SKILL_WEIGHTS.get(skill, 3)
        confidence = min(1.0, (freq * weight) / 50.0)

        if in_resume:
            if freq >= 2:
                action = "have"
            else:
                action = "demonstrate"
        else:
            if weight >= 7:
                action = "learn"
                missing.append(skill)
            elif weight >= 4:
                action = "learn"
                missing.append(skill)
            else:
                action = "skip"

        skills_list.append(SkillEntry(
            skill=skill, frequency=freq, in_resume=in_resume,
            action=action, confidence=round(confidence, 3)
        ))

    # Sort by frequency desc
    skills_list.sort(key=lambda x: (-x.frequency, -x.confidence))
    top_missing = missing[:6]

    # Explanations (UAV static stability concept from P4)
    exp_12th = (
        "Imagine a ruler balanced on your finger. If you push one end down, it tilts — "
        "that's like an aircraft pitching. The Centre of Gravity (CG) is your finger's position. "
        "The Neutral Point is where a gust of wind hits. If your CG is in front of the Neutral Point, "
        "the plane automatically corrects itself — it's stable. If CG is behind it, the plane "
        "keeps pitching and goes out of control. We measure this safety margin as Static Margin (%)."
    )
    exp_professional = (
        "Static stability is characterised by the static margin SM = (N₀ − xᴄᴳ) / ̅c, "
        "where N₀ is the neutral point, xᴄᴳ the CG location, and ̅c the mean aerodynamic chord. "
        "A positive SM (CG ahead of NP) produces a restoring pitching moment dCm/dCL < 0, ensuring "
        "the vehicle returns to trim after a disturbance. For UAVs, SM typically ranges 5–15% MAC; "
        "values outside this cause either excessive stick force or divergent phugoid response."
    )
    exp_fiveyear = (
        "You know how a toy plane has to be balanced right or it falls? If the heavy part is "
        "a little bit in front of the middle, the plane flies straight and stays safe. "
        "That's what we check — where the heavy part is!"
    )

    record = GenAISkillAnalysis(
        session_id=data.session_id,
        role_title=data.role_title,
        jd_text=data.jd_text[:2000],
        resume_text=data.resume_text[:2000],
        skills_json=json.dumps([s.model_dump() for s in skills_list]),
        gap_analysis_json=json.dumps({"missing": top_missing}),
        similarity_score=similarity_pct,
        top_missing_skills=",".join(top_missing),
    )
    db.add(record)
    db.commit()

    return SkillAnalysisResult(
        similarity_score=similarity_pct,
        skills=skills_list,
        top_missing=top_missing,
        interview_questions=INTERVIEW_QS,
        thirty_day_plan=THIRTY_DAY_PLAN,
        explanation_12th=exp_12th,
        explanation_professional=exp_professional,
        explanation_fiveyear=exp_fiveyear,
        ml_match_pct=similarity_pct,
    )


@router.get("/history")
async def get_history(db: Session = Depends(get_db)):
    rows = db.query(GenAISkillAnalysis).order_by(GenAISkillAnalysis.created_at.desc()).limit(20).all()
    return [{"id": r.id, "role": r.role_title, "similarity_pct": r.similarity_score,
             "top_missing": r.top_missing_skills,
             "created_at": r.created_at.isoformat() if r.created_at else None}
            for r in rows]
