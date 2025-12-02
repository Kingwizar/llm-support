# test_navigation.py
import os
from ddgs import DDGS

def test_duckduckgo_search(query: str):
    """
    Recherche complète DuckDuckGo (text + news + answers + general search)
    et sauvegarde dans /tmp ou C:\\tmp.
    """

    results = []
    ddg = DDGS()

    # 1️⃣ Recherche textuelle principale
    try:
        results_text = list(ddg.text(query, max_results=5))
        results.append(("TEXT", results_text))
    except Exception as e:
        results.append(("TEXT_ERROR", str(e)))

    # 2️⃣ Recherche actualités
    try:
        results_news = list(ddg.news(query, max_results=5))
        results.append(("NEWS", results_news))
    except Exception as e:
        results.append(("NEWS_ERROR", str(e)))

    # 3️⃣ Réponses directes DuckDuckGo
    try:
        results_ans = list(ddg.answers(query))
        results.append(("ANSWERS", results_ans))
    except Exception as e:
        results.append(("ANSWERS_ERROR", str(e)))

    # 4️⃣ Recherche générale fallback
    try:
        results_gen = list(ddg.search(query, max_results=5))
        results.append(("SEARCH", results_gen))
    except Exception as e:
        results.append(("SEARCH_ERROR", str(e)))

    # Formatage du fichier
    output = f"=== DuckDuckGo Search TEST ===\nQuery: {query}\n\n"

    for category, items in results:
        output += f"\n############## {category} ##############\n"
        if isinstance(items, list):
            for i, r in enumerate(items, 1):
                output += f"\n--- Résultat {i} ---\n"
                output += f"{r}\n"
        else:
            output += f"Erreur : {items}\n"

    # Sauvegarde dans tmp
    tmp_dir = "C:\\tmp" if os.name == "nt" else "/tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    file_path = os.path.join(tmp_dir, "duckduckgo_test.txt")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"📄 Fichier généré : {file_path}")
    return results


if __name__ == "__main__":
    test_duckduckgo_search("quand la CAN 2025")
    print("🔍 Terminé.")
