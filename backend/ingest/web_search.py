import requests
from bs4 import BeautifulSoup

def simple_web_search(query: str, num_results: int = 3):
    """
    Fait une recherche simple sur DuckDuckGo et renvoie les premiers résultats (titre + extrait + lien)
    """
    try:
        url = f"https://duckduckgo.com/html/?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        results = []

        for link in soup.select(".result__a")[:num_results]:
            title = link.get_text()
            href = link["href"]
            snippet_tag = link.find_parent("div", class_="result")
            snippet = snippet_tag.get_text(separator=" ", strip=True)[:300] if snippet_tag else ""
            results.append({
                "title": title,
                "url": href,
                "snippet": snippet
            })

        return results

    except Exception as e:
        return [{"error": f"Erreur recherche web : {e}"}]
