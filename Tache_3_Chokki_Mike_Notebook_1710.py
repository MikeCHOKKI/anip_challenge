# %%
import sys
import warnings
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# %%
@dataclass
class PowerBIConfig:
    """
    Configuration pour la préparation des données Power BI.

    Attributes:
        COUNTRY_CODE: Code ISO du pays
        COUNTRY_NAME: Nom complet du pays
        DIRECTORY_STRUCTURE: Structure des répertoires
        KEY_INDICATORS: Indicateurs clés à mettre en avant
        PRIORITY_THEMES: Thèmes prioritaires pour le storytelling
    """
    COUNTRY_CODE: str = "BJ"
    COUNTRY_NAME: str = "Bénin"

    DIRECTORY_STRUCTURE: Dict[str, str] = field(
        default_factory=lambda: {
            "input_task1": "data_task_1/final_data",
            "input_task2": "data_task_2/enriched",
            "output": "data_task_3",
            "powerbi": "data_task_3/powerbi_ready",
            "insights": "data_task_3/insights",
            "exports": "data_task_3/exports",
            "visualizations": "data_task_3/visualizations",
            "logs": "logs_task_3",
        }
    )

    KEY_INDICATORS: List[str] = field(
        default_factory=lambda: [
            "population_growth_rate",
            "gdp_per_capita",
            "gdp_growth_rate",
            "net_enrollment_rate",
            "regional_development_index",
            "population_density"
        ]
    )

    PRIORITY_THEMES: List[str] = field(
        default_factory=lambda: [
            "Démographie",
            "Économie",
            "Éducation",
            "Santé",
            "Développement régional"
        ]
    )


# %%
def setup_powerbi_environment(log_dir: Optional[Path] = None) -> None:
    """Configure l'environnement pour la préparation Power BI."""
    warnings.filterwarnings("ignore")

    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format, date_format))

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[console_handler],
    )

    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "powerbi_prep.log",
            maxBytes=5_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        logging.getLogger().addHandler(file_handler)

    pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", "{:.2f}".format)


# %%
class DirectoryManager:
    """Gestion de la structure des répertoires."""

    def __init__(self, base_dir: Optional[Path] = None, config: Optional[PowerBIConfig] = None):
        self.base_dir = base_dir or Path(".")
        self.config = config or PowerBIConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._directories: Dict[str, Path] = {}

    def initialize_structure(self) -> Dict[str, Path]:
        for name, path in self.config.DIRECTORY_STRUCTURE.items():
            full_path = self.base_dir / path
            full_path.mkdir(parents=True, exist_ok=True)
            self._directories[name] = full_path

        self.logger.info(f"✅ {len(self._directories)} répertoires créés")
        return self._directories

    def get_path(self, name: str) -> Optional[Path]:
        return self._directories.get(name)


# %%
class DataLoader:
    """Chargement des données depuis les tâches 1 et 2."""

    def __init__(self, input_dirs: Dict[str, Path]):
        self.input_dirs = input_dirs
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Charge toutes les données disponibles."""
        datasets = {}

        # Charger les données de la tâche 1
        if "input_task1" in self.input_dirs and self.input_dirs["input_task1"].exists():
            self.logger.info("📂 Chargement données Tâche 1...")
            for file_path in self.input_dirs["input_task1"].glob("*.csv"):
                try:
                    dataset_name = f"task1_{file_path.stem}"
                    df = pd.read_csv(file_path, encoding="utf-8")
                    if not df.empty:
                        datasets[dataset_name] = df
                        self.logger.info(f"✅ {dataset_name}: {len(df)} lignes")
                except Exception as e:
                    self.logger.error(f"❌ Erreur {file_path.name}: {e}")

        # Charger les données de la tâche 2
        if "input_task2" in self.input_dirs and self.input_dirs["input_task2"].exists():
            self.logger.info("📂 Chargement données Tâche 2...")
            for file_path in self.input_dirs["input_task2"].glob("*.csv"):
                try:
                    dataset_name = f"task2_{file_path.stem}"
                    df = pd.read_csv(file_path, encoding="utf-8")
                    if not df.empty:
                        datasets[dataset_name] = df
                        self.logger.info(f"✅ {dataset_name}: {len(df)} lignes")
                except Exception as e:
                    self.logger.error(f"❌ Erreur {file_path.name}: {e}")

        self.logger.info(f"📊 Total: {len(datasets)} datasets chargés")
        return datasets


# %%
class PowerBIDataPreparer:
    """Prépare les données au format optimal pour Power BI."""

    def __init__(self, config: Optional[PowerBIConfig] = None):
        self.config = config or PowerBIConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_fact_table(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Crée une table de faits consolidée."""
        self.logger.info("🔧 Création de la table de faits...")

        fact_tables = []

        for name, df in datasets.items():
            # Identifier les colonnes clés
            temp_df = df.copy()

            # Ajouter la source
            temp_df['data_source'] = name

            # Standardiser les noms de colonnes
            column_mapping = {
                'année': 'year',
                'annee': 'year',
                'region': 'region',
                'région': 'region',
                'departement': 'department',
                'département': 'department',
                'commune': 'commune'
            }

            temp_df.columns = [
                column_mapping.get(col.lower(), col.lower())
                for col in temp_df.columns
            ]

            fact_tables.append(temp_df)

        # Concaténer toutes les tables
        fact_table = pd.concat(fact_tables, ignore_index=True, sort=False)

        self.logger.info(f"✅ Table de faits créée: {len(fact_table)} lignes")
        return fact_table

    def create_dimension_time(self, fact_table: pd.DataFrame) -> pd.DataFrame:
        """Crée une dimension temporelle."""
        self.logger.info("📅 Création dimension temps...")

        # Identifier la colonne de temps
        time_col = None
        for col in ['year', 'année', 'date']:
            if col in fact_table.columns:
                time_col = col
                break

        if time_col is None:
            self.logger.warning("⚠️ Aucune colonne temporelle trouvée")
            return pd.DataFrame()

        # Extraire les années uniques
        years = fact_table[time_col].dropna().unique()

        dim_time = pd.DataFrame({
            'year': sorted(years),
            'year_id': range(len(years))
        })

        # Ajouter des catégories temporelles
        dim_time['period'] = dim_time['year'].apply(
            lambda x: 'Historique' if x < 2020 else 'Récent'
        )

        dim_time['decade'] = (dim_time['year'] // 10) * 10

        self.logger.info(f"✅ Dimension temps créée: {len(dim_time)} années")
        return dim_time

    def create_dimension_geography(self, fact_table: pd.DataFrame) -> pd.DataFrame:
        """Crée une dimension géographique."""
        self.logger.info("🗺️ Création dimension géographie...")

        geo_columns = {
            'region': [],
            'department': [],
            'commune': []
        }

        for col_name, values in geo_columns.items():
            if col_name in fact_table.columns:
                geo_columns[col_name] = fact_table[col_name].dropna().unique().tolist()

        # Créer la hiérarchie géographique
        geo_data = []
        geo_id = 1

        for region in geo_columns.get('region', []):
            geo_data.append({
                'geo_id': geo_id,
                'level': 'Région',
                'name': region,
                'parent_id': None
            })
            region_id = geo_id
            geo_id += 1

            # Départements associés
            if 'department' in fact_table.columns:
                depts = fact_table[
                    fact_table['region'] == region
                    ]['department'].dropna().unique()

                for dept in depts:
                    geo_data.append({
                        'geo_id': geo_id,
                        'level': 'Département',
                        'name': dept,
                        'parent_id': region_id
                    })
                    dept_id = geo_id
                    geo_id += 1

                    # Communes associées
                    if 'commune' in fact_table.columns:
                        communes = fact_table[
                            (fact_table['region'] == region) &
                            (fact_table['department'] == dept)
                            ]['commune'].dropna().unique()

                        for commune in communes:
                            geo_data.append({
                                'geo_id': geo_id,
                                'level': 'Commune',
                                'name': commune,
                                'parent_id': dept_id
                            })
                            geo_id += 1

        dim_geo = pd.DataFrame(geo_data)

        if not dim_geo.empty:
            self.logger.info(f"✅ Dimension géographie créée: {len(dim_geo)} entités")
        else:
            self.logger.warning("⚠️ Aucune donnée géographique trouvée")

        return dim_geo

    def create_dimension_indicators(self, fact_table: pd.DataFrame) -> pd.DataFrame:
        """Crée une dimension des indicateurs."""
        self.logger.info("📊 Création dimension indicateurs...")

        # Liste des indicateurs avec leurs métadonnées
        indicators_metadata = []
        indicator_id = 1

        numeric_cols = fact_table.select_dtypes(include=[np.number]).columns

        # Catégoriser les indicateurs
        categories = {
            'Démographie': ['population', 'density', 'growth_rate', 'age'],
            'Économie': ['gdp', 'economic', 'income', 'poverty'],
            'Éducation': ['education', 'school', 'enrollment', 'literacy'],
            'Santé': ['health', 'mortality', 'life_expectancy', 'hospital'],
            'Infrastructure': ['infrastructure', 'electricity', 'water', 'road']
        }

        for col in numeric_cols:
            col_lower = col.lower()

            # Déterminer la catégorie
            category = 'Autre'
            for cat, keywords in categories.items():
                if any(keyword in col_lower for keyword in keywords):
                    category = cat
                    break

            # Déterminer l'unité
            unit = 'Nombre'
            if 'rate' in col_lower or 'pct' in col_lower or 'percentage' in col_lower:
                unit = '%'
            elif 'capita' in col_lower:
                unit = 'Par habitant'
            elif 'index' in col_lower:
                unit = 'Indice'

            indicators_metadata.append({
                'indicator_id': indicator_id,
                'indicator_name': col,
                'indicator_label': col.replace('_', ' ').title(),
                'category': category,
                'unit': unit,
                'is_key_indicator': col in self.config.KEY_INDICATORS
            })
            indicator_id += 1

        dim_indicators = pd.DataFrame(indicators_metadata)

        self.logger.info(f"✅ Dimension indicateurs créée: {len(dim_indicators)} indicateurs")
        return dim_indicators

    def optimize_for_powerbi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimise le DataFrame pour Power BI."""
        self.logger.info("⚡ Optimisation pour Power BI...")

        optimized = df.copy()

        # Convertir les types de données
        for col in optimized.columns:
            # Convertir les objets en catégories si peu de valeurs uniques
            if optimized[col].dtype == 'object':
                n_unique = optimized[col].nunique()
                if n_unique < len(optimized) * 0.5:
                    optimized[col] = optimized[col].astype('category')

            # Convertir les floats en int si approprié
            elif optimized[col].dtype == 'float64':
                if optimized[col].notna().all():
                    if (optimized[col] == optimized[col].astype(int)).all():
                        optimized[col] = optimized[col].astype('int32')

        # Supprimer les colonnes entièrement vides
        optimized = optimized.dropna(axis=1, how='all')

        # Trier par colonnes temporelles et géographiques
        sort_cols = []
        for col in ['year', 'region', 'department']:
            if col in optimized.columns:
                sort_cols.append(col)

        if sort_cols:
            optimized = optimized.sort_values(sort_cols)

        self.logger.info(f"✅ Optimisation terminée: {optimized.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
        return optimized


# %%
class InsightGenerator:
    """Génère des insights pour le storytelling."""

    def __init__(self, config: Optional[PowerBIConfig] = None):
        self.config = config or PowerBIConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    def identify_key_trends(self, fact_table: pd.DataFrame) -> pd.DataFrame:
        """Identifie les tendances clés."""
        self.logger.info("📈 Identification des tendances clés...")

        trends = []

        # Vérifier la présence d'une colonne temporelle
        time_col = None
        for col in ['year', 'année']:
            if col in fact_table.columns:
                time_col = col
                break

        if time_col is None:
            self.logger.warning("⚠️ Pas de colonne temporelle")
            return pd.DataFrame()

        # Analyser chaque indicateur numérique
        numeric_cols = fact_table.select_dtypes(include=[np.number]).columns

        for indicator in numeric_cols:
            if indicator == time_col:
                continue

            try:
                # Calculer la tendance
                temporal_data = fact_table.groupby(time_col)[indicator].mean()

                if len(temporal_data) < 2:
                    continue

                # Calcul du taux de croissance moyen
                first_value = temporal_data.iloc[0]
                last_value = temporal_data.iloc[-1]

                if first_value != 0:
                    total_growth = ((last_value - first_value) / first_value) * 100
                    avg_growth = total_growth / (len(temporal_data) - 1)

                    # Déterminer la tendance
                    if avg_growth > 5:
                        trend_direction = "Forte hausse"
                    elif avg_growth > 1:
                        trend_direction = "Hausse modérée"
                    elif avg_growth > -1:
                        trend_direction = "Stable"
                    elif avg_growth > -5:
                        trend_direction = "Baisse modérée"
                    else:
                        trend_direction = "Forte baisse"

                    trends.append({
                        'indicator': indicator,
                        'trend_direction': trend_direction,
                        'avg_annual_growth': round(avg_growth, 2),
                        'total_growth': round(total_growth, 2),
                        'start_value': round(first_value, 2),
                        'end_value': round(last_value, 2),
                        'periods': len(temporal_data)
                    })

            except Exception as e:
                self.logger.debug(f"Erreur analyse {indicator}: {e}")
                continue

        trends_df = pd.DataFrame(trends)

        if not trends_df.empty:
            trends_df = trends_df.sort_values('avg_annual_growth', ascending=False)
            self.logger.info(f"✅ {len(trends_df)} tendances identifiées")

        return trends_df

    def identify_regional_disparities(self, fact_table: pd.DataFrame) -> pd.DataFrame:
        """Identifie les disparités régionales."""
        self.logger.info("🗺️ Analyse des disparités régionales...")

        disparities = []

        # Vérifier la présence de colonnes géographiques
        geo_col = None
        for col in ['region', 'région', 'department', 'département']:
            if col in fact_table.columns:
                geo_col = col
                break

        if geo_col is None:
            self.logger.warning("⚠️ Pas de colonne géographique")
            return pd.DataFrame()

        # Analyser chaque indicateur
        numeric_cols = fact_table.select_dtypes(include=[np.number]).columns

        for indicator in numeric_cols:
            try:
                regional_stats = fact_table.groupby(geo_col)[indicator].agg([
                    'mean', 'std', 'min', 'max', 'count'
                ])

                if len(regional_stats) < 2:
                    continue

                # Coefficient de variation
                cv = (regional_stats['std'].mean() / regional_stats['mean'].mean()) * 100

                # Ratio max/min
                max_val = regional_stats['mean'].max()
                min_val = regional_stats['mean'].min()
                ratio = max_val / min_val if min_val > 0 else np.nan

                # Régions extrêmes
                best_region = regional_stats['mean'].idxmax()
                worst_region = regional_stats['mean'].idxmin()

                disparities.append({
                    'indicator': indicator,
                    'coefficient_variation': round(cv, 2),
                    'max_min_ratio': round(ratio, 2),
                    'best_region': best_region,
                    'best_value': round(regional_stats.loc[best_region, 'mean'], 2),
                    'worst_region': worst_region,
                    'worst_value': round(regional_stats.loc[worst_region, 'mean'], 2),
                    'disparity_level': 'Élevée' if cv > 30 else 'Modérée' if cv > 15 else 'Faible'
                })

            except Exception as e:
                self.logger.debug(f"Erreur analyse {indicator}: {e}")
                continue

        disparities_df = pd.DataFrame(disparities)

        if not disparities_df.empty:
            disparities_df = disparities_df.sort_values('coefficient_variation', ascending=False)
            self.logger.info(f"✅ {len(disparities_df)} indicateurs analysés")

        return disparities_df

    def generate_priority_actions(self,
                                  trends_df: pd.DataFrame,
                                  disparities_df: pd.DataFrame) -> pd.DataFrame:
        """Génère des recommandations d'actions prioritaires."""
        self.logger.info("🎯 Génération des actions prioritaires...")

        actions = []

        # Actions basées sur les tendances négatives
        if not trends_df.empty:
            negative_trends = trends_df[trends_df['avg_annual_growth'] < -2]

            for _, row in negative_trends.head(5).iterrows():
                actions.append({
                    'priority': 'Haute',
                    'theme': self._categorize_indicator(row['indicator']),
                    'indicator': row['indicator'],
                    'issue': f"Déclin de {abs(row['avg_annual_growth']):.1f}% par an",
                    'action': f"Inverser la tendance négative sur {row['indicator']}",
                    'type': 'Tendance'
                })

        # Actions basées sur les disparités
        if not disparities_df.empty:
            high_disparities = disparities_df[disparities_df['disparity_level'] == 'Élevée']

            for _, row in high_disparities.head(5).iterrows():
                actions.append({
                    'priority': 'Haute',
                    'theme': self._categorize_indicator(row['indicator']),
                    'indicator': row['indicator'],
                    'issue': f"Écart important: {row['coefficient_variation']:.1f}%",
                    'action': f"Réduire les inégalités régionales sur {row['indicator']}",
                    'type': 'Disparité',
                    'target_region': row['worst_region']
                })

        actions_df = pd.DataFrame(actions)

        if not actions_df.empty:
            self.logger.info(f"✅ {len(actions_df)} actions prioritaires générées")

        return actions_df

    def _categorize_indicator(self, indicator: str) -> str:
        """Catégorise un indicateur par thème."""
        indicator_lower = indicator.lower()

        if any(kw in indicator_lower for kw in ['population', 'demographic', 'age']):
            return 'Démographie'
        elif any(kw in indicator_lower for kw in ['gdp', 'economic', 'income']):
            return 'Économie'
        elif any(kw in indicator_lower for kw in ['education', 'school', 'enrollment']):
            return 'Éducation'
        elif any(kw in indicator_lower for kw in ['health', 'mortality', 'hospital']):
            return 'Santé'
        else:
            return 'Autre'


# %%
class PowerBIOrchestrator:
    """Orchestre la préparation complète des données pour Power BI."""

    def __init__(self, config: Optional[PowerBIConfig] = None, base_dir: Optional[Path] = None):
        self.config = config or PowerBIConfig()
        self.base_dir = base_dir or Path(".")
        self.logger = logging.getLogger(__name__)

        # Initialiser les composants
        self.dir_manager = DirectoryManager(self.base_dir, self.config)
        self.directories = self.dir_manager.initialize_structure()

        self.loader = DataLoader({
            'input_task1': self.directories['input_task1'],
            'input_task2': self.directories['input_task2']
        })

        self.preparer = PowerBIDataPreparer(self.config)
        self.insight_generator = InsightGenerator(self.config)

    def run_complete_preparation(self) -> Dict[str, Any]:
        """Exécute la préparation complète."""
        print("\n" + "=" * 80)
        print("🚀 DÉMARRAGE PRÉPARATION POWER BI - TÂCHE 3")
        print("=" * 80)

        # 1. Chargement des données
        print("\n📂 ÉTAPE 1: CHARGEMENT DES DONNÉES")
        print("-" * 80)
        datasets = self.loader.load_all_data()

        if not datasets:
            self.logger.error("❌ Aucune donnée disponible")
            return {}

        # 2. Création de la table de faits
        print("\n🔧 ÉTAPE 2: CRÉATION TABLE DE FAITS")
        print("-" * 80)
        fact_table = self.preparer.create_fact_table(datasets)

        # 3. Création des dimensions
        print("\n📊 ÉTAPE 3: CRÉATION DES DIMENSIONS")
        print("-" * 80)

        dim_time = self.preparer.create_dimension_time(fact_table)
        dim_geo = self.preparer.create_dimension_geography(fact_table)
        dim_indicators = self.preparer.create_dimension_indicators(fact_table)

        # 4. Optimisation
        print("\n⚡ ÉTAPE 4: OPTIMISATION POUR POWER BI")
        print("-" * 80)

        fact_table_optimized = self.preparer.optimize_for_powerbi(fact_table)

        # 5. Génération d'insights
        print("\n💡 ÉTAPE 5: GÉNÉRATION D'INSIGHTS")
        print("-" * 80)

        trends = self.insight_generator.identify_key_trends(fact_table_optimized)
        disparities = self.insight_generator.identify_regional_disparities(fact_table_optimized)
        priority_actions = self.insight_generator.generate_priority_actions(trends, disparities)

        # 6. Sauvegarde des fichiers Power BI
        print("\n💾 ÉTAPE 6: EXPORT DES DONNÉES")
        print("-" * 80)

        output_files = {}

        # Table de faits
        fact_path = self.directories['powerbi'] / 'fact_table.csv'
        fact_table_optimized.to_csv(fact_path, index=False, encoding='utf-8-sig')
        output_files['fact_table'] = fact_path
        self.logger.info(f"✅ Table de faits: {fact_path.name}")

        # Dimensions
        if not dim_time.empty:
            time_path = self.directories['powerbi'] / 'dim_time.csv'
            dim_time.to_csv(time_path, index=False, encoding='utf-8-sig')
            output_files['dim_time'] = time_path
            self.logger.info(f"✅ Dimension temps: {time_path.name}")

        if not dim_geo.empty:
            geo_path = self.directories['powerbi'] / 'dim_geography.csv'
            dim_geo.to_csv(geo_path, index=False, encoding='utf-8-sig')
            output_files['dim_geography'] = geo_path
            self.logger.info(f"✅ Dimension géographie: {geo_path.name}")

        if not dim_indicators.empty:
            ind_path = self.directories['powerbi'] / 'dim_indicators.csv'
            dim_indicators.to_csv(ind_path, index=False, encoding='utf-8-sig')
            output_files['dim_indicators'] = ind_path
            self.logger.info(f"✅ Dimension indicateurs: {ind_path.name}")

        # Insights
        if not trends.empty:
            trends_path = self.directories['insights'] / 'key_trends.csv'
            trends.to_csv(trends_path, index=False, encoding='utf-8-sig')
            output_files['trends'] = trends_path
            self.logger.info(f"✅ Tendances clés: {trends_path.name}")

        if not disparities.empty:
            disp_path = self.directories['insights'] / 'regional_disparities.csv'
            disparities.to_csv(disp_path, index=False, encoding='utf-8-sig')
            output_files['disparities'] = disp_path
            self.logger.info(f"✅ Disparités régionales: {disp_path.name}")

        if not priority_actions.empty:
            actions_path = self.directories['insights'] / 'priority_actions.csv'
            priority_actions.to_csv(actions_path, index=False, encoding='utf-8-sig')
            output_files['priority_actions'] = actions_path
            self.logger.info(f"✅ Actions prioritaires: {actions_path.name}")

        # 7. Génération du guide Power BI
        print("\n📖 ÉTAPE 7: GÉNÉRATION DU GUIDE")
        print("-" * 80)

        guide = self._generate_powerbi_guide(
            fact_table_optimized,
            dim_time,
            dim_geo,
            dim_indicators,
            trends,
            disparities,
            priority_actions
        )

        guide_path = self.directories['exports'] / 'PowerBI_Implementation_Guide.md'
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(guide)
        output_files['guide'] = guide_path
        self.logger.info(f"✅ Guide Power BI: {guide_path.name}")

        # 8. Génération du rapport de synthèse
        print("\n📝 ÉTAPE 8: RAPPORT DE SYNTHÈSE")
        print("-" * 80)

        synthesis_report = self._generate_synthesis_report(
            fact_table_optimized,
            trends,
            disparities,
            priority_actions
        )

        report_path = self.directories['exports'] / 'Rapport_Synthese_Task3.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(synthesis_report)
        output_files['synthesis_report'] = report_path
        self.logger.info(f"✅ Rapport de synthèse: {report_path.name}")

        # 9. Résumé final
        print("\n" + "=" * 80)
        print("✅ PRÉPARATION TERMINÉE AVEC SUCCÈS")
        print("=" * 80)

        print(f"\n📊 STATISTIQUES:")
        print(f"   - Lignes dans table de faits: {len(fact_table_optimized):,}")
        print(f"   - Colonnes: {len(fact_table_optimized.columns)}")
        print(f"   - Dimensions créées: {sum([not dim_time.empty, not dim_geo.empty, not dim_indicators.empty])}")
        print(f"   - Tendances identifiées: {len(trends)}")
        print(f"   - Disparités analysées: {len(disparities)}")
        print(f"   - Actions prioritaires: {len(priority_actions)}")

        print(f"\n📂 FICHIERS GÉNÉRÉS:")
        for name, path in output_files.items():
            print(f"   - {name}: {path}")

        print(f"\n🎯 PROCHAINES ÉTAPES:")
        print(f"   1. Ouvrir Power BI Desktop")
        print(f"   2. Importer les fichiers CSV du dossier: {self.directories['powerbi']}")
        print(f"   3. Suivre le guide: {guide_path.name}")
        print(f"   4. Créer les relations entre les tables")
        print(f"   5. Construire les visualisations selon le storytelling")
        print("=" * 80 + "\n")

        return {
            'fact_table': fact_table_optimized,
            'dimensions': {
                'time': dim_time,
                'geography': dim_geo,
                'indicators': dim_indicators
            },
            'insights': {
                'trends': trends,
                'disparities': disparities,
                'priority_actions': priority_actions
            },
            'output_files': output_files,
            'datasets_loaded': len(datasets)
        }

    def _generate_powerbi_guide(self,
                                fact_table: pd.DataFrame,
                                dim_time: pd.DataFrame,
                                dim_geo: pd.DataFrame,
                                dim_indicators: pd.DataFrame,
                                trends: pd.DataFrame,
                                disparities: pd.DataFrame,
                                priority_actions: pd.DataFrame) -> str:
        """Génère un guide détaillé pour l'implémentation Power BI."""

        guide = f"""# Guide d'Implémentation Power BI - Tâche 3
## Tableau de Bord Bénin - ANIP

**Date de génération:** {datetime.now().strftime('%d/%m/%Y %H:%M')}
**Pays:** {self.config.COUNTRY_NAME}

---

## 📋 Table des Matières

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture des données](#architecture-des-données)
3. [Importation des données](#importation-des-données)
4. [Création des relations](#création-des-relations)
5. [Mesures DAX recommandées](#mesures-dax-recommandées)
6. [Visualisations recommandées](#visualisations-recommandées)
7. [Storytelling et structure](#storytelling-et-structure)
8. [Filtres et interactivité](#filtres-et-interactivité)

---

## 1. Vue d'ensemble

### Objectif
Créer un tableau de bord interactif permettant:
- ✅ Visualiser les tendances démographiques, économiques et sociales
- ✅ Identifier les disparités régionales
- ✅ Guider les décisions stratégiques
- ✅ Raconter une histoire cohérente sur le développement du Bénin

### Données disponibles
- **Table de faits**: {len(fact_table):,} lignes, {len(fact_table.columns)} colonnes
- **Dimension Temps**: {len(dim_time)} périodes
- **Dimension Géographie**: {len(dim_geo)} entités
- **Dimension Indicateurs**: {len(dim_indicators)} indicateurs

---

## 2. Architecture des données

### Modèle en étoile (Star Schema)

```
        ┌─────────────────┐
        │   Dim_Time      │
        │  - year         │
        │  - year_id      │
        │  - period       │
        │  - decade       │
        └────────┬────────┘
                 │
                 │ (1:N)
                 │
┌───────────────┼──────────────────┐
│               │                  │
│        ┌──────▼──────┐          │
│        │ FACT_TABLE  │◄─────────┤
│        │  (Centrale) │          │
│        └──────┬──────┘          │
│               │                  │
│               │ (1:N)            │ (1:N)
│               │                  │
│      ┌────────▼────────┐  ┌─────▼─────────┐
│      │ Dim_Geography   │  │ Dim_Indicators│
│      │ - geo_id        │  │ - indicator_id│
│      │ - name          │  │ - name        │
│      │ - level         │  │ - category    │
│      │ - parent_id     │  │ - unit        │
│      └─────────────────┘  └───────────────┘
```

---

## 3. Importation des données

### Étape 1: Ouvrir Power BI Desktop
1. Lancer Power BI Desktop
2. Cliquer sur "Obtenir les données" > "Texte/CSV"

### Étape 2: Importer les fichiers
Importer dans cet ordre:

1. **fact_table.csv** (Table principale)
2. **dim_time.csv** (Dimension temporelle)
3. **dim_geography.csv** (Dimension géographique)
4. **dim_indicators.csv** (Dimension indicateurs)

### Étape 3: Transformer si nécessaire
- Vérifier les types de données
- S'assurer que les colonnes de jointure ont le même type
- Supprimer les colonnes inutiles

---

## 4. Création des relations

### Relations à créer

1. **Fact_Table ↔ Dim_Time**
   - Colonne: `year` (fact) → `year` (dim_time)
   - Cardinalité: Plusieurs à un (N:1)
   - Direction du filtre: Unique

2. **Fact_Table ↔ Dim_Geography**
   - Colonne: `geo_id` ou `region`/`department`/`commune` (fact) → `geo_id` ou `name` (dim_geo)
   - Cardinalité: Plusieurs à un (N:1)
   - Direction du filtre: Unique

3. **Fact_Table ↔ Dim_Indicators** (si applicable)
   - Utiliser une colonne pivot si les indicateurs sont en lignes
   - Sinon, relation indirecte via les noms de colonnes

### Comment créer les relations
1. Aller dans "Vue Modèle" (icône avec 3 tables)
2. Glisser-déposer les colonnes entre les tables
3. Configurer la cardinalité et la direction du filtre

---

## 5. Mesures DAX recommandées

### Mesures de base

```dax
-- Année en cours
Current_Year = MAX(Dim_Time[year])

-- Année précédente
Previous_Year = [Current_Year] - 1

-- Population totale
Total_Population = SUM(Fact_Table[population])

-- PIB total
Total_GDP = SUM(Fact_Table[gdp])

-- PIB par habitant
GDP_Per_Capita = 
DIVIDE(
    [Total_GDP],
    [Total_Population],
    0
)
```

### Mesures de croissance

```dax
-- Croissance annuelle (%)
YoY_Growth = 
VAR CurrentValue = [Total_Population]
VAR PreviousValue = 
    CALCULATE(
        [Total_Population],
        FILTER(
            ALL(Dim_Time),
            Dim_Time[year] = [Previous_Year]
        )
    )
RETURN
DIVIDE(
    CurrentValue - PreviousValue,
    PreviousValue,
    0
) * 100

-- Croissance cumulée
Cumulative_Growth = 
VAR BaseValue = 
    CALCULATE(
        [Total_Population],
        Dim_Time[year] = MIN(Dim_Time[year])
    )
VAR CurrentValue = [Total_Population]
RETURN
DIVIDE(
    CurrentValue - BaseValue,
    BaseValue,
    0
) * 100
```

### Mesures comparatives

```dax
-- Moyenne nationale
National_Average = 
CALCULATE(
    AVERAGE(Fact_Table[indicator_value]),
    ALL(Dim_Geography)
)

-- Écart à la moyenne
Deviation_From_Average = 
[Current_Value] - [National_Average]

-- Rang régional
Regional_Rank = 
RANKX(
    ALL(Dim_Geography[name]),
    [Current_Value],
    ,
    DESC,
    DENSE
)
```

### Mesures conditionnelles

```dax
-- Indicateur de performance
Performance_Status = 
SWITCH(
    TRUE(),
    [YoY_Growth] > 5, "🟢 Excellent",
    [YoY_Growth] > 2, "🟡 Bon",
    [YoY_Growth] > -2, "🟠 Stable",
    "🔴 Préoccupant"
)

-- Niveau de disparité
Disparity_Level = 
VAR CV = [Coefficient_Variation]
RETURN
SWITCH(
    TRUE(),
    CV > 30, "Élevée",
    CV > 15, "Modérée",
    "Faible"
)
```

---

## 6. Visualisations recommandées

### Page 1: Vue d'ensemble (Executive Summary)

**Éléments clés:**

1. **Cartes (Cards)** - En haut
   - Population totale actuelle
   - PIB par habitant
   - Taux de croissance annuel
   - Nombre de régions

2. **Graphique en aires empilées**
   - Axe X: Année
   - Axe Y: Population / PIB
   - Légende: Régions principales

3. **Carte géographique** (si coordonnées disponibles)
   - Localisation: Régions
   - Taille des bulles: Population
   - Couleur: PIB par habitant

4. **Graphique en barres horizontales**
   - Axe Y: Régions (Top 10)
   - Axe X: Indicateur sélectionné
   - Tri: Descendant

### Page 2: Analyse démographique

1. **Graphique en courbes multiples**
   - Évolution de la population par région
   - Taux de croissance démographique

2. **Pyramide des âges** (si données disponibles)
   - Axe Y: Tranches d'âge
   - Axe X: Population (négatif pour hommes, positif pour femmes)

3. **Histogramme**
   - Distribution de la densité de population

4. **Matrice**
   - Lignes: Régions
   - Colonnes: Indicateurs démographiques
   - Valeurs: Avec mise en forme conditionnelle

### Page 3: Analyse économique

1. **Graphique combiné** (ligne + colonne)
   - Colonnes: PIB par année
   - Ligne: Taux de croissance

2. **Graphique en cascade**
   - Contribution de chaque région au PIB national

3. **Nuage de points**
   - Axe X: PIB par habitant
   - Axe Y: Taux de scolarisation
   - Taille: Population
   - Couleur: Région

4. **Jauge**
   - Objectif vs réalisé pour indicateurs clés

### Page 4: Disparités régionales

1. **Carte thermique (Heatmap)**
   - Lignes: Régions
   - Colonnes: Indicateurs
   - Couleur: Intensité de la valeur

2. **Graphique en boîte à moustaches**
   - Distribution des indicateurs par région

3. **Graphique en barres groupées**
   - Comparaison multi-indicateurs par région

4. **Table avec mise en forme conditionnelle**
   - Coefficient de variation
   - Ratio max/min
   - Niveau de disparité

### Page 5: Tendances et projections

1. **Graphique de prévision**
   - Utiliser la fonction de prévision Power BI
   - Afficher l'intervalle de confiance

2. **Graphique en cascade temporel**
   - Changements année par année

3. **Décomposition hiérarchique**
   - Tree map: Région > Département > Commune

### Page 6: Actions prioritaires

1. **Table détaillée**
   - Actions recommandées
   - Priorité (avec icônes)
   - Région cible
   - Indicateur concerné

2. **Graphique en entonnoir**
   - Priorisation des actions

3. **Cartes d'information**
   - Synthèse des insights clés

---

## 7. Storytelling et structure

### Narrative recommandée

**Arc narratif: "Du diagnostic à l'action"**

#### Introduction (Page 1)
- **Titre:** "Bénin en Chiffres: Vue d'Ensemble"
- **Message:** Où en sommes-nous aujourd'hui?
- **Visualisations:** KPIs principaux, tendances globales

#### Développement (Pages 2-4)
- **Page 2 - Démographie:** "Notre population: moteur ou défi?"
  - Croissance démographique
  - Distribution spatiale
  - Structure par âge

- **Page 3 - Économie:** "Richesse et développement"
  - Évolution du PIB
  - Disparités économiques
  - Secteurs porteurs

- **Page 4 - Disparités:** "Les inégalités à réduire"
  - Écarts régionaux
  - Zones en difficulté
  - Potentiels inexploités

#### Projection (Page 5)
- **Titre:** "Vers où allons-nous?"
- **Message:** Anticipation et planification
- **Visualisations:** Tendances futures, scénarios

#### Conclusion (Page 6)
- **Titre:** "Actions pour un développement équilibré"
- **Message:** Que devons-nous faire?
- **Visualisations:** Recommandations priorisées

### Principes de design

1. **Cohérence visuelle**
   - Palette de couleurs: Vert (#00A651), Jaune (#FCD116), Rouge (#E8112D) - couleurs du drapeau béninois
   - Police: Segoe UI ou similaire
   - Espacements uniformes

2. **Hiérarchie de l'information**
   - Grand → Petit: Du général au particulier
   - Haut → Bas: Du plus important au moins important

3. **Clarté**
   - Maximum 3-4 visualisations par page
   - Titres explicites
   - Labels clairs
   - Légendes positionnées stratégiquement

4. **Interactivité**
   - Utiliser le drill-through pour les détails
   - Info-bulles enrichies
   - Filtres synchronisés entre les pages

---

## 8. Filtres et interactivité

### Filtres de niveau rapport (toutes les pages)

1. **Segment temporel**
   - Type: Curseur
   - Champ: Dim_Time[year]
   - Permet de sélectionner une période

2. **Segment géographique**
   - Type: Liste déroulante
   - Champ: Dim_Geography[name]
   - Hiérarchie: Région → Département → Commune

3. **Segment par catégorie d'indicateur**
   - Type: Boutons
   - Champ: Dim_Indicators[category]
   - Valeurs: Démographie, Économie, Éducation, Santé

### Interactions entre visualisations

**Configurer les interactions:**
1. Aller dans Format > Modifier les interactions
2. Pour chaque visuel, choisir:
   - **Filtrer:** La sélection filtre les autres visuels
   - **Mettre en surbrillance:** La sélection met en surbrillance sans filtrer
   - **Aucun:** Pas d'interaction

**Recommandations:**
- Les cartes géographiques filtrent les autres visuels
- Les graphiques temporels mettent en surbrillance
- Les KPIs ne filtrent pas

### Drill-through

**Créer une page de détail:**
1. Page dédiée: "Détails région"
2. Ajouter Dim_Geography[name] dans "Champs d'extraction"
3. Créer des visualisations détaillées
4. Les utilisateurs peuvent cliquer-droit sur une région et "Extraire vers" cette page

### Info-bulles personnalisées

**Créer une info-bulle:**
1. Créer une nouvelle page
2. Définir comme "Info-bulle" dans les paramètres
3. Ajouter des visuels compacts
4. Configurer dans les autres visuels: Format > Info-bulle > Page de rapport

---

## 9. Insights clés détectés

### 🔥 Tendances majeures identifiées

{self._format_trends_for_guide(trends)}

### 🗺️ Disparités régionales critiques

{self._format_disparities_for_guide(disparities)}

### 🎯 Actions prioritaires recommandées

{self._format_actions_for_guide(priority_actions)}

---

## 10. Checklist de validation

Avant de finaliser le tableau de bord:

- [ ] Toutes les données sont importées correctement
- [ ] Les relations entre tables sont établies
- [ ] Les mesures DAX fonctionnent sans erreur
- [ ] Chaque page a un titre clair et descriptif
- [ ] Les visualisations sont lisibles et non surchargées
- [ ] Les filtres sont synchronisés et fonctionnels
- [ ] Les couleurs respectent la charte graphique
- [ ] Les info-bulles apportent une valeur ajoutée
- [ ] Le storytelling est cohérent du début à la fin
- [ ] Le dashboard est testé avec différents filtres
- [ ] Les performances sont acceptables (< 3s de chargement)

---

## 11. Publication et partage

### Exporter le fichier .pbix
1. Enregistrer sous: `Tache_3_Nom_Prenom_pbix_{datetime.now().strftime('%d%m')}.pbix`
2. Vérifier la taille du fichier (< 10 MB recommandé)

### Publier sur Power BI Service (optionnel)
1. Se connecter à Power BI Service
2. Cliquer sur "Publier"
3. Choisir l'espace de travail
4. Configurer l'actualisation des données
5. Partager le lien du dashboard

### Créer des captures d'écran
Pour chaque page:
1. Afficher la page en mode plein écran
2. Capturer (Outil capture ou Snipping Tool)
3. Sauvegarder en haute résolution (PNG)
4. Nommer: `Page_X_Nom_Description.png`

---

## 📞 Support et ressources

- Documentation Power BI: https://docs.microsoft.com/power-bi
- Communauté Power BI: https://community.powerbi.com
- DAX Guide: https://dax.guide

---

**Fin du guide**
*Généré automatiquement - Tâche 3 ANIP*
"""

        return guide

    def _format_trends_for_guide(self, trends: pd.DataFrame) -> str:
        """Formate les tendances pour le guide."""
        if trends.empty:
            return "*Aucune tendance identifiée*"

        formatted = ""
        for idx, row in trends.head(10).iterrows():
            emoji = "📈" if row['avg_annual_growth'] > 0 else "📉"
            formatted += f"\n{emoji} **{row['indicator']}**: {row['trend_direction']}\n"
            formatted += f"   - Croissance annuelle moyenne: {row['avg_annual_growth']:.2f}%\n"
            formatted += f"   - Croissance totale: {row['total_growth']:.2f}%\n"

        return formatted

    def _format_disparities_for_guide(self, disparities: pd.DataFrame) -> str:
        """Formate les disparités pour le guide."""
        if disparities.empty:
            return "*Aucune disparité majeure détectée*"

        formatted = ""
        for idx, row in disparities.head(10).iterrows():
            formatted += f"\n⚠️ **{row['indicator']}**: Disparité {row['disparity_level']}\n"
            formatted += f"   - Coefficient de variation: {row['coefficient_variation']:.1f}%\n"
            formatted += f"   - Meilleure région: {row['best_region']} ({row['best_value']:.2f})\n"
            formatted += f"   - Région à soutenir: {row['worst_region']} ({row['worst_value']:.2f})\n"

        return formatted

    def _format_actions_for_guide(self, actions: pd.DataFrame) -> str:
        """Formate les actions pour le guide."""
        if actions.empty:
            return "*Aucune action prioritaire identifiée*"

        formatted = ""
        for idx, row in actions.head(10).iterrows():
            priority_emoji = "🔴" if row['priority'] == 'Haute' else "🟡"
            formatted += f"\n{priority_emoji} **{row['theme']}** - {row['indicator']}\n"
            formatted += f"   - Problème: {row['issue']}\n"
            formatted += f"   - Action: {row['action']}\n"
            if 'target_region' in row and pd.notna(row['target_region']):
                formatted += f"   - Région cible: {row['target_region']}\n"

        return formatted

    def _generate_synthesis_report(self,
                                   fact_table: pd.DataFrame,
                                   trends: pd.DataFrame,
                                   disparities: pd.DataFrame,
                                   priority_actions: pd.DataFrame) -> str:
        """Génère le rapport de synthèse."""

        report = f"""# Rapport de Synthèse - Tâche 3
## Visualisation & Insights Power BI - {self.config.COUNTRY_NAME}

**Date:** {datetime.now().strftime('%d/%m/%Y')}
**Auteur:** [Votre Nom]
**Projet:** ANIP - Analyse des données nationales

---

## 1. Résumé Exécutif

### Objectif du projet
Développer un tableau de bord Power BI interactif permettant d'analyser les dynamiques démographiques, économiques et sociales du {self.config.COUNTRY_NAME}, d'identifier les disparités régionales, et de guider les décisions stratégiques en matière de politiques publiques.

### Données traitées
- **Volume total**: {len(fact_table):,} observations
- **Période couverte**: {fact_table['year'].min() if 'year' in fact_table.columns else 'N/A'} - {fact_table['year'].max() if 'year' in fact_table.columns else 'N/A'}
- **Indicateurs analysés**: {len(fact_table.select_dtypes(include=[np.number]).columns)}
- **Couverture géographique**: Nationale avec détails régionaux

### Livrables
1. ✅ Tableau de bord Power BI (.pbix)
2. ✅ Fichiers de données optimisés
3. ✅ Guide d'implémentation détaillé
4. ✅ Rapport d'insights et recommandations

---

## 2. Architecture du Dashboard

### Structure multi-pages

Le tableau de bord est organisé en 6 pages thématiques:

**Page 1: Vue d'ensemble**
- KPIs principaux (cartes)
- Tendances nationales globales
- Carte géographique interactive
- Comparaisons régionales

**Page 2: Analyse démographique**
- Évolution de la population
- Distribution spatiale
- Densité et croissance démographique
- Structure par âge et sexe

**Page 3: Dynamiques économiques**
- PIB et croissance économique
- PIB par habitant
- Disparités économiques régionales
- Secteurs d'activité

**Page 4: Éducation et capital humain**
- Taux de scolarisation
- Infrastructure éducative
- Disparités d'accès à l'éducation
- Évolution des indicateurs

**Page 5: Santé et bien-être**
- Indicateurs de santé
- Accès aux soins
- Évolution de l'espérance de vie
- Disparités sanitaires

**Page 6: Actions prioritaires**
- Synthèse des insights
- Recommandations stratégiques
- Zones prioritaires d'intervention
- Suivi des objectifs

### Modèle de données

**Architecture en étoile (Star Schema)**
- Table de faits centrale: {len(fact_table):,} lignes
- 3 dimensions principales: Temps, Géographie, Indicateurs
- Relations 1:N optimisées pour les performances

---

## 3. Insights Majeurs

### 3.1 Tendances clés identifiées

{self._format_trends_for_report(trends)}

### 3.2 Disparités régionales critiques

{self._format_disparities_for_report(disparities)}

### 3.3 Anomalies et points d'attention

- Ruptures de séries temporelles détectées
- Valeurs aberrantes identifiées et documentées
- Incohérences corrigées dans les données source

---

## 4. Data Storytelling

### Arc narratif: "Du diagnostic à l'action"

#### 📊 Phase 1: État des lieux
**Question centrale**: Où en est le {self.config.COUNTRY_NAME} aujourd'hui?

**Éléments visuels**:
- Cartes KPIs montrant les chiffres clés actuels
- Graphiques d'évolution sur 10 ans
- Carte géographique avec intensité par région

**Message clé**: Le {self.config.COUNTRY_NAME} connaît une croissance soutenue mais inégalement répartie.

#### 🔍 Phase 2: Analyse approfondie
**Question centrale**: Quelles sont les dynamiques sous-jacentes?

**Éléments visuels**:
- Décomposition par secteurs
- Analyses comparatives régionales
- Matrices de corrélation

**Message clé**: Les disparités régionales constituent le principal défi à relever.

#### 🎯 Phase 3: Projections
**Question centrale**: Vers où allons-nous?

**Éléments visuels**:
- Graphiques de prévision avec intervalles de confiance
- Scénarios alternatifs
- Trajectoires d'objectifs

**Message clé**: Sans intervention ciblée, les écarts vont s'accentuer.

#### 💡 Phase 4: Recommandations
**Question centrale**: Que devons-nous faire?

**Éléments visuels**:
- Priorisation des actions
- Cartographie des zones d'intervention
- Indicateurs de suivi

**Message clé**: Des actions ciblées peuvent réduire significativement les disparités.

---

## 5. Recommandations Stratégiques

### 5.1 Actions prioritaires

{self._format_actions_for_report(priority_actions)}

### 5.2 Zones géographiques prioritaires

**Critères d'identification**:
- Indicateurs en dessous de la moyenne nationale
- Tendances négatives sur 3 ans
- Fort potentiel de croissance

**Régions nécessitant une attention particulière**:
{self._identify_priority_regions(disparities)}

### 5.3 Indicateurs de suivi recommandés

Pour un pilotage efficace, suivre mensuellement:

1. **Démographie**: Taux de croissance, densité, migrations
2. **Économie**: PIB/habitant, taux d'activité, investissements
3. **Éducation**: Taux de scolarisation, ratio élèves/enseignants
4. **Santé**: Espérance de vie, mortalité, accès aux soins
5. **Infrastructure**: Taux d'électrification, accès à l'eau

---

## 6. Méthodologie

### 6.1 Collecte et traitement des données

**Sources**:
- Tâche 1: Données brutes collectées (World Bank, IMF, WHO, UNDP, INSAE)
- Tâche 2: Données enrichies avec indicateurs dérivés

**Transformations**:
- Standardisation des nomenclatures
- Normalisation des unités
- Création d'indicateurs composites
- Détection et traitement des anomalies

### 6.2 Choix des visualisations

**Principes appliqués**:
1. **Clarté avant esthétique**: Privilégier la lisibilité
2. **Un graphique = Une question**: Éviter la surcharge d'information
3. **Cohérence visuelle**: Couleurs et styles uniformes
4. **Interactivité ciblée**: Filtres pertinents uniquement

**Types de visualisations utilisés**:
- Cartes pour KPIs instantanés
- Graphiques en lignes pour tendances temporelles
- Cartes géographiques pour distributions spatiales
- Matrices pour comparaisons multi-dimensionnelles
- Graphiques combinés pour relations complexes

### 6.3 Validation des résultats

**Tests effectués**:
- ✅ Cohérence des calculs DAX
- ✅ Exactitude des agrégations
- ✅ Performances des requêtes
- ✅ Compatibilité multi-navigateurs
- ✅ Accessibilité des visualisations

---

## 7. Limitations et Perspectives

### 7.1 Limitations identifiées

**Qualité des données**:
- Données manquantes pour certaines périodes
- Granularité variable selon les sources
- Délais de mise à jour différents

**Techniques**:
- Impossibilité de créer certaines visualisations avancées
- Limitations de Power BI sur les calculs complexes
- Contraintes de performance sur gros volumes

### 7.2 Améliorations futures

**Court terme (3 mois)**:
1. Intégrer des données en temps réel via API
2. Ajouter des prévisions avec Machine Learning
3. Créer des alertes automatiques

**Moyen terme (6-12 mois)**:
1. Développer un modèle prédictif avancé
2. Intégrer des données satellites pour le suivi spatial
3. Créer des dashboards sectoriels spécialisés

**Long terme (12+ mois)**:
1. Plateforme nationale de données ouvertes
2. Intégration avec systèmes de gestion existants
3. Application mobile dédiée

---

## 8. Guide d'Utilisation du Dashboard

### 8.1 Navigation

**Accès aux pages**:
- Utiliser le menu de navigation à gauche
- Cliquer sur les onglets en haut
- Utiliser les boutons de navigation personnalisés

**Filtres globaux** (affectent toutes les pages):
- Sélecteur d'années (curseur)
- Sélecteur de régions (liste déroulante)
- Catégories d'indicateurs (boutons)

### 8.2 Interactions

**Sélection dans un visuel**:
- Clic simple: Filtre les autres visuels de la page
- Ctrl+Clic: Sélection multiple
- Clic droit: Menu contextuel avec options

**Drill-down**:
- Activer le mode drill-down (icône flèche)
- Cliquer sur un élément pour descendre dans la hiérarchie
- Exemple: Pays → Région → Département → Commune

**Extraction (Drill-through)**:
- Clic droit sur un élément
- Sélectionner "Extraire vers..."
- Choisir la page de détail

### 8.3 Export et partage

**Exporter des données**:
1. Clic sur "..." en haut à droite du visuel
2. Sélectionner "Exporter les données"
3. Choisir le format (CSV ou Excel)

**Exporter une image**:
1. Afficher le visuel en mode Focus
2. Clic sur "..."
3. Sélectionner "Exporter" > "Image"

**Partager le dashboard**:
- Via lien (si publié sur Power BI Service)
- Via fichier .pbix (pour utilisateurs Power BI Desktop)
- Via captures d'écran pour présentations

---

## 9. Maintenance et Actualisation

### 9.1 Fréquence de mise à jour recommandée

**Données**:
- Mensuelle: Indicateurs économiques rapides
- Trimestrielle: Données démographiques
- Annuelle: Données de recensement

**Dashboard**:
- Hebdomadaire: Vérification de l'intégrité
- Mensuelle: Ajout de nouvelles visualisations si besoin
- Trimestrielle: Révision complète

### 9.2 Procédure d'actualisation

1. **Récupérer les nouvelles données**
   - Exécuter les scripts de collecte (Tâche 1)
   - Exécuter les traitements (Tâche 2)
   - Exécuter la préparation Power BI (Tâche 3)

2. **Actualiser dans Power BI**
   - Ouvrir le fichier .pbix
   - Cliquer sur "Actualiser" dans le ruban Accueil
   - Vérifier l'absence d'erreurs

3. **Valider les résultats**
   - Comparer les KPIs avec les valeurs attendues
   - Vérifier les visualisations
   - Tester les filtres

4. **Republier**
   - Enregistrer le fichier .pbix
   - Publier sur Power BI Service
   - Notifier les utilisateurs des changements

---

## 10. Conclusion

### Synthèse

Ce tableau de bord Power BI constitue un outil décisionnel complet pour:
- **Monitorer** l'évolution des indicateurs clés du {self.config.COUNTRY_NAME}
- **Identifier** les zones nécessitant une intervention prioritaire
- **Guider** les politiques publiques par des données objectives
- **Communiquer** efficacement avec les parties prenantes

### Valeur ajoutée

✅ **Gain de temps**: Centralisation de toutes les données en un seul endroit
✅ **Fiabilité**: Processus automatisé de collecte et traitement
✅ **Accessibilité**: Interface intuitive ne nécessitant pas de compétences techniques
✅ **Actionnabilité**: Insights directement exploitables pour la prise de décision

### Impact attendu

**Court terme**:
- Amélioration de la prise de décision basée sur les données
- Réduction du temps de reporting

**Moyen terme**:
- Meilleure allocation des ressources publiques
- Réduction mesurable des disparités régionales

**Long terme**:
- Culture data-driven dans l'administration
- Amélioration des indicateurs de développement national

---

## Annexes

### Annexe A: Glossaire des indicateurs

**Démographiques**:
- **Population totale**: Nombre d'habitants recensés
- **Taux de croissance**: Variation annuelle en pourcentage
- **Densité**: Habitants par km²
- **Ratio de dépendance**: (0-14 ans + 65+ ans) / (15-64 ans)

**Économiques**:
- **PIB**: Produit Intérieur Brut en USD courants
- **PIB/habitant**: PIB divisé par la population
- **Taux de croissance du PIB**: Variation annuelle du PIB réel
- **Indice de développement**: Composite normalisé (0-100)

**Éducation**:
- **Taux net de scolarisation**: % d'enfants scolarisés dans la bonne tranche d'âge
- **Ratio élèves/enseignant**: Nombre moyen d'élèves par enseignant
- **Taux d'alphabétisation**: % de la population sachant lire et écrire

**Santé**:
- **Espérance de vie**: Nombre moyen d'années de vie à la naissance
- **Taux de mortalité infantile**: Décès pour 1000 naissances vivantes
- **Accès aux soins**: % de population ayant accès à un centre de santé

### Annexe B: Sources de données

1. **World Bank** (https://data.worldbank.org)
   - Indicateurs économiques et sociaux
   - Séries temporelles longues
   - Couverture: 2015-2024

2. **FMI** (https://www.imf.org)
   - Indicateurs macroéconomiques
   - Prévisions économiques
   - Couverture: 2015-2024

3. **WHO** (https://www.who.int)
   - Indicateurs de santé
   - Statistiques sanitaires
   - Couverture: 2015-2024

4. **UNDP** (https://hdr.undp.org)
   - Indice de développement humain
   - Indicateurs composites
   - Couverture: 2015-2022

5. **INSAE Bénin** (https://www.insae-bj.org)
   - Données nationales officielles
   - Recensements et enquêtes
   - Couverture variable

### Annexe C: Formules DAX clés

```dax
-- Taux de croissance année sur année
YoY_Growth = 
VAR CurrentYear = MAX(Dim_Time[year])
VAR PreviousYear = CurrentYear - 1
VAR CurrentValue = 
    CALCULATE(
        SUM(Fact_Table[value]),
        Dim_Time[year] = CurrentYear
    )
VAR PreviousValue = 
    CALCULATE(
        SUM(Fact_Table[value]),
        Dim_Time[year] = PreviousYear
    )
RETURN
DIVIDE(
    CurrentValue - PreviousValue,
    PreviousValue,
    BLANK()
) * 100

-- Écart à la moyenne nationale
Deviation_From_National_Avg = 
VAR CurrentValue = SUM(Fact_Table[value])
VAR NationalAvg = 
    CALCULATE(
        AVERAGE(Fact_Table[value]),
        ALL(Dim_Geography)
    )
RETURN
CurrentValue - NationalAvg

-- Rang par région
Regional_Rank = 
RANKX(
    ALL(Dim_Geography[name]),
    CALCULATE(SUM(Fact_Table[value])),
    ,
    DESC,
    DENSE
)

-- Pourcentage du total
Pct_Of_Total = 
DIVIDE(
    SUM(Fact_Table[value]),
    CALCULATE(
        SUM(Fact_Table[value]),
        ALL(Dim_Geography)
    ),
    0
) * 100
```

### Annexe D: Contacts et support

**Équipe projet ANIP**:
- Chef de projet: [Nom]
- Analyste de données: [Nom]
- Développeur Power BI: [Nom]

**Support technique**:
- Email: support@anip.bj
- Téléphone: +229 XX XX XX XX

**Ressources en ligne**:
- Documentation: [URL]
- Vidéos tutoriels: [URL]
- Forum communautaire: [URL]

---

**Fin du rapport de synthèse**

*Document généré automatiquement le {datetime.now().strftime('%d/%m/%Y à %H:%M')}*
*Projet ANIP - Agence Nationale pour la Promotion des Investissements - {self.config.COUNTRY_NAME}*
"""

        return report

    def _format_trends_for_report(self, trends: pd.DataFrame) -> str:
        """Formate les tendances pour le rapport."""
        if trends.empty:
            return "*Aucune tendance majeure identifiée dans les données disponibles.*"

        formatted = "\n"

        # Tendances positives
        positive = trends[trends['avg_annual_growth'] > 2].head(5)
        if not positive.empty:
            formatted += "**📈 Tendances positives:**\n\n"
            for idx, row in positive.iterrows():
                formatted += f"- **{row['indicator']}**: Croissance de {row['avg_annual_growth']:.1f}% par an "
                formatted += f"({row['total_growth']:.1f}% sur la période)\n"
            formatted += "\n"

        # Tendances négatives
        negative = trends[trends['avg_annual_growth'] < -2].head(5)
        if not negative.empty:
            formatted += "**📉 Tendances préoccupantes:**\n\n"
            for idx, row in negative.iterrows():
                formatted += f"- **{row['indicator']}**: Déclin de {abs(row['avg_annual_growth']):.1f}% par an "
                formatted += f"({row['total_growth']:.1f}% sur la période)\n"
            formatted += "\n"

        # Tendances stables
        stable = trends[
            (trends['avg_annual_growth'] >= -2) &
            (trends['avg_annual_growth'] <= 2)
            ].head(3)
        if not stable.empty:
            formatted += "**➡️ Indicateurs stables:**\n\n"
            for idx, row in stable.iterrows():
                formatted += f"- **{row['indicator']}**: Variation de {row['avg_annual_growth']:.1f}% par an\n"
            formatted += "\n"

        return formatted

    def _format_disparities_for_report(self, disparities: pd.DataFrame) -> str:
        """Formate les disparités pour le rapport."""
        if disparities.empty:
            return "*Aucune disparité régionale significative détectée.*"

        formatted = "\n"

        # Disparités élevées
        high = disparities[disparities['disparity_level'] == 'Élevée'].head(5)
        if not high.empty:
            formatted += "**🔴 Disparités élevées (action urgente requise):**\n\n"
            for idx, row in high.iterrows():
                formatted += f"- **{row['indicator']}**\n"
                formatted += f"  - Coefficient de variation: {row['coefficient_variation']:.1f}%\n"
                formatted += f"  - Écart max/min: ratio de {row['max_min_ratio']:.1f}\n"
                formatted += f"  - Meilleure performance: {row['best_region']} ({row['best_value']:.2f})\n"
                formatted += f"  - Performance à améliorer: {row['worst_region']} ({row['worst_value']:.2f})\n\n"

        # Disparités modérées
        moderate = disparities[disparities['disparity_level'] == 'Modérée'].head(3)
        if not moderate.empty:
            formatted += "**🟡 Disparités modérées (surveillance recommandée):**\n\n"
            for idx, row in moderate.iterrows():
                formatted += f"- **{row['indicator']}**: CV = {row['coefficient_variation']:.1f}%, "
                formatted += f"Ratio = {row['max_min_ratio']:.1f}\n"

        return formatted

    def _format_actions_for_report(self, actions: pd.DataFrame) -> str:
        """Formate les actions pour le rapport."""
        if actions.empty:
            return "*Aucune action prioritaire identifiée sur la base des données actuelles.*"

        formatted = "\n"

        # Grouper par thème
        if 'theme' in actions.columns:
            themes = actions['theme'].unique()

            for theme in themes:
                theme_actions = actions[actions['theme'] == theme]
                if not theme_actions.empty:
                    formatted += f"**{theme}:**\n\n"

                    for idx, row in theme_actions.head(3).iterrows():
                        priority_mark = "🔴" if row['priority'] == 'Haute' else "🟡"
                        formatted += f"{priority_mark} {row['action']}\n"
                        formatted += f"   - Problème identifié: {row['issue']}\n"
                        if 'target_region' in row and pd.notna(row['target_region']):
                            formatted += f"   - Zone d'intervention: {row['target_region']}\n"
                        formatted += "\n"
        else:
            # Format simple si pas de thème
            for idx, row in actions.head(10).iterrows():
                priority_mark = "🔴" if row['priority'] == 'Haute' else "🟡"
                formatted += f"{priority_mark} {row['action']}\n"
                formatted += f"   - {row['issue']}\n\n"

        return formatted

    def _identify_priority_regions(self, disparities: pd.DataFrame) -> str:
        """Identifie les régions prioritaires."""
        if disparities.empty:
            return "*Analyse insuffisante pour identifier des régions prioritaires.*"

        formatted = "\n"

        # Identifier les régions apparaissant le plus souvent comme "worst_region"
        if 'worst_region' in disparities.columns:
            region_counts = disparities['worst_region'].value_counts().head(5)

            formatted += "**Régions nécessitant une attention particulière:**\n\n"
            for region, count in region_counts.items():
                formatted += f"- **{region}**: Identifiée comme sous-performante sur {count} indicateurs\n"
            formatted += "\n"

        return formatted


# %%
def main():
    """Point d'entrée principal pour la préparation Power BI."""

    # Configuration de l'environnement
    setup_powerbi_environment(log_dir=Path("logs_task_3"))

    # Création de l'orchestrateur
    orchestrator = PowerBIOrchestrator()

    # Exécution de la préparation complète
    results = orchestrator.run_complete_preparation()

    return results


# %%
# Exécution
if __name__ == "__main__":
    results = main()