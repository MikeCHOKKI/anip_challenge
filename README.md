# Challenge ANIP : données socio-économiques du Bénin

Pipeline Python en trois tâches, réalisé en septembre et octobre 2025 pour le challenge ANIP : collecter des
données publiques sur le Bénin, les nettoyer, les analyser, puis les préparer pour un tableau de bord Power BI.

Chaque tâche existe en deux versions identiques : le notebook Jupyter (`.ipynb`) et son export en script
Python (`.py`).

## Les trois tâches

| Tâche | Fichiers | Ce qu'elle fait | Sortie |
|---|---|---|---|
| 1. Collecte et nettoyage | `Tache_1_Chokki_Mike_Notebook_1710` | Interroge les API de la Banque mondiale, du FMI, de l'OMS, du PNUD et de l'UNESCO, OpenStreetMap (Overpass) et les publications de l'INStaD ; nettoie et harmonise les séries de 2015 à 2024, avec un rapport de nettoyage et des mesures de performance | `data_task_1/` |
| 2. Analyse | `Tache_2_Chokki_Mike_Notebook_1710` | Statistiques descriptives, tendances, analyse spatiale, corrélations, détection d'anomalies et indicateurs composites | `data_task_2/` |
| 3. Préparation Power BI | `Tache_3_Chokki_Mike_Notebook_1710` | Tables prêtes pour Power BI, génération d'enseignements et documentation du tableau de bord | `data_task_3/` |

Chaque tâche lit la sortie de la précédente : il faut les exécuter dans l'ordre.

## Documentation

- [`glossaire_tache_1.md`](glossaire_tache_1.md) : variables, collecteurs et structures de la tâche 1.
- [`glossaire_anomalie_methode_tache_1.md`](glossaire_anomalie_methode_tache_1.md) : anomalies relevées et
  choix méthodologiques de la collecte.
- [`glossaire_tache_2.md`](glossaire_tache_2.md) : anomalies et choix méthodologiques de l'analyse.

## Lancer le projet

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt jupyter
jupyter notebook
```

Ouvrir puis exécuter les notebooks dans l'ordre 1, 2, 3. La tâche 3 peut aussi se lancer en script :
`python Tache_3_Chokki_Mike_Notebook_1710.py`.

La tâche 1 a besoin d'Internet : elle appelle des API publiques, sans clé. Les dossiers de données, d'exports
et de journaux sont créés à l'exécution et ne sont pas versionnés.

## Licence

[MIT](LICENSE) — © 2025 Mike E.S. Chokki
