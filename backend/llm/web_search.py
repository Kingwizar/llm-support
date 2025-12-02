# llm/web_search.py
import os
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS   # 🔥 nouvelle API DuckDuckGo
from datetime import datetime


# ================================
# 🔍 Extraction HTML complète
# ================================
def extract_full_text_from_url(url: str, max_chars: int = 3000) -> str:
    """Télécharge et extrait le texte lisible d'une page web."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Nettoyage
        for tag in soup(["script", "style", "noscript", "footer", "header", "nav"]):
            tag.extract()

        text = " ".join(soup.stripped_strings)
        return text[:max_chars]

    except Exception as e:
        return f"[Erreur d'extraction: {e}]"


# ================================
# 🦆 Recherche DuckDuckGo (texte)
# ================================
def duckduckgo_text_search(query: str, num_results: int = 5):
    """Recherche textuelle DuckDuckGo."""
    try:
        with DDGS() as ddg:
            results = list(ddg.text(query, max_results=num_results))
            return results
    except Exception as e:
        return [{"error": f"DuckDuckGo TEXT error: {e}"}]


# ================================
# 📰 Recherche DuckDuckGo News
# ================================
def duckduckgo_news_search(query: str, num_results: int = 5):
    """Recherche actualités DuckDuckGo."""
    try:
        with DDGS() as ddg:
            results = list(ddg.news(query, max_results=num_results))
            return results
    except Exception as e:
        return [{"error": f"DuckDuckGo NEWS error: {e}"}]


# ================================
# 🔎 Fallback "search" général
# ================================
def duckduckgo_general_search(query: str, num_results: int = 5):
    """Fallback general search DuckDuckGo."""
    try:
        with DDGS() as ddg:
            results = list(ddg.search(query, max_results=num_results))
            return results
    except Exception as e:
        return [{"error": f"DuckDuckGo SEARCH error: {e}"}]


# ================================
# 🌐 Fonction principale
# ================================
def simple_web_search(query: str, num_results: int = 5, full_content: bool = True):
    """
    Recherche Internet via DuckDuckGo :
    - TEXT
    - NEWS
    - SEARCH fallback
    + extraction HTML optionnelle
    + sauvegarde debug dans /tmp
    """

    print(f"[INFO] Recherche DuckDuckGo pour : {query}")

    results = {
        "TEXT": duckduckgo_text_search(query, num_results),
        "NEWS": duckduckgo_news_search(query, num_results),
        "SEARCH": duckduckgo_general_search(query, num_results),
    }

    enriched_results = []

    for category, items in results.items():
        for item in items:
            if "error" in item:
                enriched_results.append({
                    "category": category,
                    "title": None,
                    "url": None,
                    "snippet": item["error"],
                    "content": ""
                })
                continue

            title = item.get("title") or item.get("article_title") or ""
            url   = item.get("href") or item.get("url") or ""
            body  = item.get("body") or item.get("excerpt") or ""

            content = extract_full_text_from_url(url) if (full_content and url) else ""

            enriched_results.append({
                "category": category,
                "title": title,
                "url": url,
                "snippet": body,
                "content": content
            })

    # ====== Génération d’un fichier debug ======
    tmp_dir = "C:\\tmp" if os.name == "nt" else "/tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    debug_path = os.path.join(tmp_dir, "websearch_debug.txt")

    with open(debug_path, "w", encoding="utf-8") as f:
        f.write(f"=== DEBUG WEBSEARCH ({datetime.now().isoformat()}) ===\n")
        f.write(f"Query: {query}\n\n")

        for r in enriched_results:
            f.write(f"--- {r['category']} ---\n")
            f.write(f"Title: {r['title']}\n")
            f.write(f"URL: {r['url']}\n")
            f.write(f"Snippet: {r['snippet']}\n\n")
            f.write(f"CONTENT:\n{r['content'][:1000]}\n")
            f.write("\n-----------------------\n")

    print(f"[DEBUG] Fichier généré : {debug_path}")

    return enriched_results
