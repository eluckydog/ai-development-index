"""
AIDI Period Search Engine v1
逐期自动搜索AI发展数据 — 仅用免费API
用法: python engine/period_search.py --start 2024-12-01 --end 2024-12-16 --period 2024-12-01
"""

import json, os, sys, time, re, subprocess
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

ARXIV_CATEGORIES = ["cs.AI", "cs.CL", "cs.LG", "stat.ML", "cs.CV"]

MODEL_KEYWORDS = [
    "GPT-4", "GPT-5", "GPT-5.5", "GPT-5.6",
    "Claude", "Opus", "Sonnet", "Haiku", "Mythos", "Capybara",
    "Gemini", "Gemma", "DeepSeek", "Llama", "LLaMA", "Grok",
    "Mistral", "Qwen", "GLM", "Kimi", "MiMo",
    "Sora", "DALL-E", "o1", "o3", "o4",
    "Stable Diffusion", "Midjourney",
]
BENCHMARK_TERMS = [
    "MMLU", "GPQA", "SWE-bench", "HumanEval",
    "Arena", "Elo", "MATH", "AIME", "GSM8K", "ARC", "AGI",
]


def search_arxiv(start_date, end_date, max_results=30):
    """Search arXiv for AI papers within date range. Free API, no auth."""
    # arXiv API supports date range in query:
    # YYYYMMDDHHMMSS format
    start_ts = start_date.replace("-", "") + "0000"
    end_ts = end_date.replace("-", "") + "2359"
    
    categories = "+OR+".join([f"cat:{c}" for c in ARXIV_CATEGORIES])
    date_filter = f"submittedDate:[{start_ts}+TO+{end_ts}]"
    keywords = "(language+model+OR+artificial+intelligence+OR+deep+learning+OR+transformer+OR+benchmark+OR+GPT+OR+LLM+OR+foundation+model)"
    
    query = f"({categories})+AND+{keywords}+AND+{date_filter}"
    url = (f"https://export.arxiv.org/api/query"
           f"?search_query={query}"
           f"&start=0&max_results={max_results}"
           f"&sortBy=submittedDate&sortOrder=descending")

    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "AIDI/1.0"})
        resp = urllib.request.urlopen(req, timeout=30)
        xml_data = resp.read().decode("utf-8")

        root = ET.fromstring(xml_data)
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

        papers = []
        for entry in root.findall("atom:entry", ns):
            pub_el = entry.find("atom:published", ns)
            pub_date_str = ""
            if pub_el is not None and pub_el.text:
                pub_date_str = pub_el.text[:10]
            elif entry.find("atom:updated", ns) is not None and entry.find("atom:updated", ns).text:
                pub_date_str = entry.find("atom:updated", ns).text[:10]

            title = entry.find("atom:title", ns)
            summary = entry.find("atom:summary", ns)
            paper_id = entry.find("atom:id", ns)
            authors = entry.findall("atom:author", ns)

            title_text = (title.text or "").strip().replace("\n", " ") if title is not None else ""
            summary_text = (summary.text or "").strip().replace("\n", " ") if summary is not None else ""
            paper_url = (paper_id.text or "").strip() if paper_id is not None else ""

            author_names = []
            for a in authors[:5]:
                an = a.find("atom:name", ns)
                if an is not None and an.text:
                    author_names.append(an.text)

            text_lower = (title_text + " " + summary_text).lower()
            relevance = sum(1 for kw in MODEL_KEYWORDS + BENCHMARK_TERMS if kw.lower() in text_lower)
            if "language model" in text_lower or "foundation model" in text_lower:
                relevance += 2

            if relevance > 0:
                papers.append({
                    "title": title_text[:200],
                    "authors": author_names,
                    "published": pub_date_str,
                    "url": paper_url,
                    "abstract": summary_text[:500],
                    "relevance": relevance,
                })

        papers.sort(key=lambda x: x["relevance"], reverse=True)
        return papers[:10]
    except Exception as e:
        return [{"error": f"arxiv: {str(e)}"}]


def search_github_trending(start_date, end_date):
    """Search GitHub for trending AI repos using gh CLI."""
    results = []
    queries = ["large-language-model", "deep-learning", "transformer", "generative-ai"]

    for q in queries:
        try:
            result = subprocess.run(
                ["gh", "api", "search/repositories",
                 "-f", f"q={q}+created:{start_date}..{end_date}",
                 "-f", "sort=stars", "-f", "order=desc", "-f", "per_page=3"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                for item in data.get("items", [])[:3]:
                    results.append({
                        "name": item["full_name"],
                        "stars": item["stargazers_count"],
                        "description": (item.get("description") or "")[:200],
                        "created": item["created_at"][:10],
                        "url": item["html_url"],
                        "language": item.get("language", "") or "",
                    })
            time.sleep(0.3)
        except Exception:
            pass

    return results


def parse_benchmarks(text):
    """Extract benchmark scores from text."""
    scores = {}
    patterns = [
        (r"MMLU[:\s]*([0-9]+\.?[0-9]*)", "MMLU"),
        (r"GPQA[:\s]*([0-9]+\.?[0-9]*)", "GPQA"),
        (r"SWE[-\s]?bench[:\s]*([0-9]+\.?[0-9]*)", "SWE-bench"),
        (r"HumanEval[:\s]*([0-9]+\.?[0-9]*)", "HumanEval"),
        (r"MATH[:\s]*([0-9]+\.?[0-9]*)", "MATH"),
        (r"AIME[:\s]*([0-9]+\.?[0-9]*)", "AIME"),
    ]
    for pat, name in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            scores[name] = float(m.group(1))
    return scores


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--period", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"[AIDI Search] Period: {args.period} ({args.start} ~ {args.end})")

    # 1. arXiv
    print("  Searching arXiv...")
    arxiv = search_arxiv(args.start, args.end)
    print(f"    Found {len(arxiv)} papers")
    for p in arxiv[:3]:
        if "error" not in p:
            print(f"    - {p['title'][:70]} ({p['published']})")

    # 2. GitHub
    print("  Searching GitHub...")
    github = search_github_trending(args.start, args.end)
    print(f"    Found {len(github)} repos")

    # 3. Activity scoring
    model_counts = {}
    bench_total = {}
    for p in arxiv:
        if "error" in p:
            continue
        text = (p.get("title", "") + " " + p.get("abstract", "")).lower()
        for kw in MODEL_KEYWORDS:
            if kw.lower() in text:
                model_counts[kw] = model_counts.get(kw, 0) + 1
        bench_total.update(parse_benchmarks(text))

    hype_models = sorted(model_counts, key=model_counts.get, reverse=True)[:5]
    model_act = min(len(model_counts) * 12, 100)
    research_act = min(len(arxiv) * 8, 100)
    open_act = min(sum(r["stars"] for r in github) / 100, 100)

    key_events = []
    for m in hype_models:
        if any(k.lower() in m.lower() for k in ["gpt", "deepseek", "claude", "gemini", "llama", "grok", "qwen", "glm", "kimi", "sora", "o1", "o3", "capybara", "mythos"]):
            key_events.append(m)

    result = {
        "period": {"start": args.start, "end": args.end, "name": args.period},
        "searched_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": ["arxiv", "github"],
        "summary": {
            "arxiv_papers": len(arxiv),
            "github_repos": len(github),
            "model_activity": model_act,
            "research_activity": research_act,
            "open_source_activity": open_act,
            "hype_models": hype_models,
        },
        "arxiv": [{"title": p["title"], "authors": p.get("authors", [])[:3],
                    "published": p["published"], "url": p["url"],
                    "abstract": p.get("abstract", "")[:300]}
                   for p in arxiv if "error" not in p],
        "github": github,
        "benchmarks_found": bench_total,
    }
    if key_events:
        result["summary"]["key_events"] = key_events[:3]

    # Save
    out = args.output or f"data/raw/{args.period}/_search.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"  Saved: {out}")
    print(f"  Activity: model={model_act} research={research_act} opensource={open_act}")
    return result


if __name__ == "__main__":
    main()
