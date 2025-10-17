# Glossaire des Variables et Termes Techniques

## 📋 Table des Matières
1. [Configuration Globale](#configuration-globale)
2. [Collecte de Données](#collecte-de-données)
3. [Nettoyage et Traitement](#nettoyage-et-traitement)
4. [Métriques et Rapports](#métriques-et-rapports)
5. [Structures de Données](#structures-de-données)

---

## 🌍 Configuration Globale

### **COUNTRY_CODE**
- **Type**: String
- **Valeur par défaut**: "BJ"
- **Description**: Code ISO 3166-1 alpha-2 du pays (Bénin)
- **Utilisation**: Filtrage des données par pays dans les API

### **COUNTRY_NAME**
- **Type**: String
- **Valeur par défaut**: "Bénin"
- **Description**: Nom complet du pays cible
- **Utilisation**: Requêtes de recherche et documentation

### **START_YEAR / END_YEAR**
- **Type**: Integer
- **Valeur par défaut**: 2015 / 2024
- **Description**: Période temporelle pour la collecte de données
- **Validation**: Doit être entre 1900 et 2100

### **REQUEST_TIMEOUT**
- **Type**: Integer (secondes)
- **Valeur par défaut**: 30
- **Description**: Délai maximal d'attente pour une requête HTTP
- **Impact**: Évite les blocages sur des serveurs lents

### **RETRY_COUNT**
- **Type**: Integer
- **Valeur par défaut**: 3
- **Description**: Nombre de tentatives en cas d'échec de requête
- **Stratégie**: Backoff exponentiel (2^tentative secondes)

### **DELAY_BETWEEN_REQUESTS**
- **Type**: Float (secondes)
- **Valeur par défaut**: 0.5
- **Description**: Pause entre requêtes successives
- **Objectif**: Respecter les limites de taux des API

---

## 📡 Collecte de Données

### **indicator_code**
- **Type**: String
- **Source**: API Banque Mondiale, FMI, OMS
- **Exemples**: "SP.POP.TOTL", "NY.GDP.MKTP.CD"
- **Description**: Identifiant unique d'un indicateur économique/social

### **indicator_name**
- **Type**: String
- **Description**: Libellé descriptif de l'indicateur
- **Exemple**: "Population, total", "PIB ($ US courants)"

### **value**
- **Type**: Numeric (Float)
- **Description**: Valeur numérique de l'indicateur pour une année donnée
- **Traitement**: Conversion automatique avec gestion des erreurs

### **year**
- **Type**: Integer
- **Description**: Année de référence de la donnée
- **Validation**: Entre START_YEAR et END_YEAR

### **source**
- **Type**: String
- **Valeurs possibles**: "World Bank API", "IMF", "WHO GHO", "INSAE", etc.
- **Description**: Provenance des données
- **Utilisation**: Traçabilité et validation croisée

### **collection_timestamp**
- **Type**: Datetime
- **Format**: ISO 8601
- **Description**: Date et heure de collecte des données
- **Ajout automatique**: Lors de la sauvegarde

---

## 🧹 Nettoyage et Traitement

### **initial_rows / final_rows**
- **Type**: Integer
- **Description**: Nombre de lignes avant/après nettoyage
- **Calcul dérivé**: rows_removed = initial_rows - final_rows

### **duplicates_removed**
- **Type**: Integer
- **Description**: Nombre de lignes dupliquées supprimées
- **Méthode**: pandas.DataFrame.drop_duplicates()

### **nulls_handled**
- **Type**: Integer
- **Description**: Nombre de valeurs manquantes traitées
- **Stratégies**: Suppression de colonnes/lignes, imputation

### **outliers_removed**
- **Type**: Integer
- **Description**: Nombre de valeurs aberrantes détectées
- **Méthode**: IQR (Interquartile Range)
- **Formule**: Q1 - 1.5×IQR < valeur < Q3 + 1.5×IQR

### **null_threshold**
- **Type**: Float (0-1)
- **Valeur par défaut**: 0.7
- **Description**: Seuil de suppression des colonnes (70% de valeurs manquantes)

### **columns_standardized**
- **Type**: List[String]
- **Description**: Colonnes dont les noms ont été normalisés
- **Transformations**: 
  - Minuscules
  - Suppression caractères spéciaux
  - Remplacement espaces par underscores

### **data_types_converted**
- **Type**: Dict[String, String]
- **Format**: {"nom_colonne": "type_origine -> type_final"}
- **Conversions courantes**: 
  - object → numeric
  - object → datetime

---

## 📊 Métriques et Rapports

### **operation_name**
- **Type**: String
- **Description**: Nom de l'opération tracée
- **Format**: "Classe.méthode" (ex: "WorldBank.fetch_indicator")

### **start_time / end_time**
- **Type**: Datetime
- **Description**: Horodatage de début/fin d'opération

### **duration_seconds**
- **Type**: Float
- **Description**: Durée d'exécution en secondes
- **Calcul**: (end_time - start_time).total_seconds()

### **items_processed**
- **Type**: Integer
- **Description**: Nombre d'éléments traités (lignes, requêtes, fichiers)

### **throughput**
- **Type**: Float (items/seconde)
- **Calcul**: items_processed / duration_seconds
- **Utilisation**: Évaluation des performances

### **success**
- **Type**: Boolean
- **Description**: Indicateur de succès/échec de l'opération
- **Valeurs**: True = succès, False = échec

### **error_message**
- **Type**: String (optionnel)
- **Description**: Message d'erreur en cas d'échec
- **Utilisation**: Débogage et traçabilité

---

## 🗺️ Structures de Données

### **DIRECTORY_STRUCTURE**
- **Type**: Dict[String, String]
- **Clés**: "data", "raw", "processed", "final_data", "logs", "exports", "docs"
- **Description**: Organisation des dossiers du projet

### **DEFAULT_WB_INDICATORS**
- **Type**: List[String]
- **Description**: Indicateurs Banque Mondiale par défaut
- **Exemples**:
  - SP.POP.TOTL: Population totale
  - NY.GDP.MKTP.CD: PIB ($ US courants)
  - SE.PRM.NENR: Taux net de scolarisation primaire

### **DEFAULT_IMF_INDICATORS**
- **Type**: List[String]
- **Description**: Indicateurs FMI par défaut
- **Exemples**:
  - NGDP_R: PIB réel
  - PCPIPCH: Inflation (%)
  - LUR: Taux de chômage

### **DEFAULT_HEALTH_INDICATORS**
- **Type**: List[String]
- **Description**: Indicateurs santé (OMS) par défaut
- **Exemples**:
  - WHOSIS_000001: Espérance de vie
  - MDG_0000000001: Mortalité infantile

### **OSM_ADMIN_LEVELS**
- **Type**: Dict[String, String]
- **Description**: Niveaux administratifs OpenStreetMap
- **Valeurs**:
  - pays: "2"
  - département: "4"
  - commune: "6"

---

## 🔍 Variables Géographiques

### **latitude / longitude**
- **Type**: Float
- **Plage valide**: [-90, 90] / [-180, 180]
- **Description**: Coordonnées géographiques
- **Source**: OpenStreetMap

### **osm_id**
- **Type**: Integer
- **Description**: Identifiant unique OpenStreetMap
- **Utilisation**: Référencement géographique

### **place_type**
- **Type**: String
- **Valeurs**: "city", "town", "village"
- **Description**: Type de localité

### **admin_level**
- **Type**: String
- **Description**: Niveau administratif OSM
- **Utilisation**: Hiérarchie territoriale

### **population**
- **Type**: Integer
- **Source**: OpenStreetMap tags
- **Description**: Population de la localité
- **Note**: Peut être manquant

---

## 📈 Indicateurs de Qualité

### **memory_mb**
- **Type**: Float
- **Unité**: Mégaoctets
- **Calcul**: df.memory_usage(deep=True).sum() / (1024²)
- **Utilisation**: Optimisation des ressources

### **null_pct**
- **Type**: Float (pourcentage)
- **Calcul**: (valeurs_null / total_valeurs) × 100
- **Seuil critique**: > 50%

### **removal_percentage**
- **Type**: Float (pourcentage)
- **Calcul**: (rows_removed / initial_rows) × 100
- **Interprétation**: Taux de suppression lors du nettoyage

### **issues_detected**
- **Type**: List[String]
- **Description**: Liste des problèmes identifiés
- **Exemples**: 
  - "Column 'x' dropped: 75% missing values"
  - "Duplicates: 150"
  - "Invalid years: 23"

---

## 🔧 Variables Techniques

### **format_type**
- **Type**: String
- **Valeurs**: "csv", "excel", "json", "parquet"
- **Description**: Format de sauvegarde des données

### **add_metadata**
- **Type**: Boolean
- **Description**: Ajouter colonnes de métadonnées
- **Colonnes ajoutées**: collection_timestamp, collector

### **max_tables**
- **Type**: Integer
- **Valeur par défaut**: 10
- **Description**: Nombre maximum de tables HTML à extraire

### **text_content_token_limit**
- **Type**: Integer (optionnel)
- **Description**: Limite de tokens pour le contenu texte
- **Usage**: Optimisation mémoire

---

## 📝 Conventions de Nommage

### **Fichiers Raw**
- **Format**: `{source}_data.csv`
- **Exemple**: `world_bank_data.csv`

### **Fichiers Cleaned**
- **Format**: `{source}_cleaned.csv`
- **Exemple**: `world_bank_cleaned.csv`

### **Fichiers Final**
- **Format**: `{category}_indicators.csv`
- **Exemples**: 
  - `economic_indicators.csv`
  - `health_indicators.csv`

### **Rapports**
- **collection_summary.csv**: Vue d'ensemble de la collecte
- **cleaning_summary.csv**: Détails du nettoyage
- **data_dictionary.csv**: Dictionnaire des variables
- **performance_metrics.csv**: Métriques de performance

---

## 🎯 Bonnes Pratiques

### **Gestion des Erreurs**
- Toujours utiliser `pd.to_numeric(..., errors='coerce')`
- Valider les URLs avant requêtes
- Logger tous les échecs avec contexte

### **Validation des Données**
- Vérifier les plages de dates (1900-2025)
- Valider coordonnées géographiques
- Détecter valeurs négatives inappropriées

### **Performance**
- Utiliser `track_progress` pour opérations longues
- Ajouter `@timer` aux méthodes critiques
- Limiter taille mémoire avec `text_content_token_limit`

### **Documentation**
- Toujours remplir attribut `source`
- Ajouter `collection_timestamp`
- Maintenir traçabilité complète

---

## 📚 Références

- **API Banque Mondiale**: https://api.worldbank.org/v2
- **API FMI**: https://www.imf.org/external/datamapper/api/v1
- **API OMS**: https://ghoapi.azureedge.net/api
- **OpenStreetMap**: https://overpass-api.de/api/interpreter
- **UNDP HDR**: https://hdr.undp.org

---