import os
import pkg_resources

# Seuil en Mo pour considérer qu'un package est "lourd"
THRESHOLD_MB = 50  

def folder_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total

print("🔍 Recherche des packages lourds...\n")

heavy = []

for dist in pkg_resources.working_set:
    pkg_name = dist.project_name
    pkg_path = dist.location + "/" + pkg_name.replace("-", "_")

    if not os.path.exists(pkg_path):
        continue

    size_bytes = folder_size(pkg_path)
    size_mb = size_bytes / (1024 * 1024)

    if size_mb >= THRESHOLD_MB:
        heavy.append((pkg_name, size_mb, pkg_path))

# Trier du plus lourd au plus léger
heavy.sort(key=lambda x: x[1], reverse=True)

if not heavy:
    print("✅ Aucun package lourd trouvé !")
else:
    print("📦 Packages lourds détectés :\n")
    for name, size, path in heavy:
        print(f"➡️ {name} — {size:.1f} MB\n    📁 {path}\n")
