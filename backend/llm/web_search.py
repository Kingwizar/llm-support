# llm/web_search.py
import os, requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()
# === Configuration Google API ===
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY_env")
GOOGLE_CX = os.getenv("GOOGLE_CX_env")

def google_search(query: str, num_results: int = 3):
    """Recherche via Google Custom Search API."""
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        return [{"error": "Clé API Google manquante. Configure GOOGLE_API_KEY et GOOGLE_CX."}]

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CX,
        "q": query,
        "num": num_results,
        "hl": "fr"
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("items", []):
            results.append({
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet", "")
            })
        return results
    except Exception as e:
        return [{"error": str(e)}]


def extract_full_text_from_url(url: str, max_chars: int = 3000) -> str:
    """Télécharge et extrait le texte lisible d'une page web."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Supprime les scripts, styles, etc.
        for tag in soup(["script", "style", "noscript", "footer", "header", "nav"]):
            tag.extract()

        text = " ".join(soup.stripped_strings)
        text = text.replace("\n", " ").strip()

        # Tronque pour éviter un prompt trop long
        return text[:max_chars]
    except Exception as e:
        return f"[Erreur d'extraction: {e}]"


def simple_web_search(query: str, num_results: int = 3, full_content: bool = True):
    """
    Recherche sur Internet via Google Custom Search + extraction complète du texte.
    Retourne les résultats avec le contenu complet de chaque site.
    """
    print(f"[INFO] Recherche Google pour: {query}")
    results = google_search(query, num_results=num_results)
    enriched = []

    for r in results:
        if "error" in r:
            enriched.append(r)
            continue
        url = r.get("url")
        snippet = r.get("snippet", "")
        content = ""
        if full_content and url:
            content = extract_full_text_from_url(url)
        enriched.append({
            "title": r.get("title"),
            "url": url,
            "snippet": snippet,
            "content": content
        })
    return enriched
