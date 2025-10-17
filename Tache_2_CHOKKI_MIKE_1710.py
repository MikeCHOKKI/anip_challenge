# %%
import functools
import sys
import warnings
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any, Union
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import zscore, pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")


# %%
@dataclass
class AnalysisConfig:
    """Configuration pour l'analyse exploratoire"""

    COUNTRY_CODE: str = "BJ"
    COUNTRY_NAME: str = "Bénin"

    # Seuils pour la détection d'anomalies
    ZSCORE_THRESHOLD: float = 3.0
    IQR_MULTIPLIER: float = 1.5
    CORRELATION_THRESHOLD: float = 0.7

    # Configuration des indicateurs
    POPULATION_AGE_YOUNG: int = 15
    POPULATION_AGE_OLD: int = 65
    GDP_BASE_YEAR: int = 2015

    # Répertoires
    DIRECTORY_STRUCTURE: Dict[str, str] = field(
        default_factory=lambda: {
            "input": "data_task_1/final_data",
            "output": "data_task_2",
            "processed": "data_task_2/processed",
            "enriched": "data_task_2/enriched",
            "analysis": "data_task_2/analysis",
            "anomalies": "data_task_2/anomalies",
            "visualizations": "data_task_2/visualizations",
            "logs": "logs_task_2",
        }
    )


# %%
def setup_analysis_environment(log_dir: Optional[Path] = None) -> None:
    """Configure l'environnement d'analyse"""
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
            log_dir / "analysis.log",
            maxBytes=5_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        logging.getLogger().addHandler(file_handler)

    pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", "{:.3f}".format)

    sns.set_theme(style="whitegrid", palette="Set2")
    plt.rcParams.update(
        {
            "figure.figsize": (14, 8),
            "axes.titlesize": 14,
            "axes.labelsize": 12,
        }
    )


# %%
class DirectoryManager:
    """Gestionnaire de répertoires pour l'analyse"""

    def __init__(
        self, base_dir: Optional[Path] = None, config: Optional[AnalysisConfig] = None
    ):
        self.base_dir = base_dir or Path(".")
        self.config = config or AnalysisConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._directories: Dict[str, Path] = {}

    def initialize_structure(self) -> Dict[str, Path]:
        """Crée la structure de répertoires"""
        for name, path in self.config.DIRECTORY_STRUCTURE.items():
            full_path = self.base_dir / path
            full_path.mkdir(parents=True, exist_ok=True)
            self._directories[name] = full_path

        self.logger.info(f"✅ {len(self._directories)} répertoires créés")
        return self._directories

    def get_path(self, name: str) -> Optional[Path]:
        """Récupère un chemin de répertoire"""
        return self._directories.get(name)


# %%
@dataclass
class AnomalyReport:
    """Rapport de détection d'anomalies"""

    dataset_name: str
    variable: str
    anomaly_type: str
    anomaly_count: int
    anomaly_percentage: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "variable": self.variable,
            "anomaly_type": self.anomaly_type,
            "count": self.anomaly_count,
            "percentage": round(self.anomaly_percentage, 2),
            "details": str(self.details),
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        }


# %%
class DataLoader:
    """Chargeur de données consolidées"""

    def __init__(self, input_dir: Path):
        self.input_dir = input_dir
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Charge tous les datasets disponibles"""
        datasets = {}

        if not self.input_dir.exists():
            self.logger.error(f"❌ Répertoire introuvable: {self.input_dir}")
            return datasets

        csv_files = list(self.input_dir.glob("*.csv"))

        if not csv_files:
            self.logger.warning(f"⚠️ Aucun fichier CSV trouvé dans {self.input_dir}")
            return datasets

        self.logger.info(f"📂 Chargement de {len(csv_files)} fichiers...")

        for file_path in csv_files:
            try:
                dataset_name = file_path.stem
                df = pd.read_csv(file_path, encoding="utf-8")

                if not df.empty:
                    datasets[dataset_name] = df
                    self.logger.info(
                        f"✅ {dataset_name}: {len(df)} lignes, {len(df.columns)} colonnes"
                    )
                else:
                    self.logger.warning(f"⚠️ {dataset_name}: fichier vide")

            except Exception as e:
                self.logger.error(
                    f"❌ Erreur lors du chargement de {file_path.name}: {e}"
                )

        self.logger.info(f"📊 Total: {len(datasets)} datasets chargés")
        return datasets


# %%
class DescriptiveAnalyzer:
    """Analyseur descriptif des données"""

    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    def analyze_dataset(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Analyse descriptive complète d'un dataset"""
        self.logger.info(f"📊 Analyse descriptive: {dataset_name}")

        analysis = {
            "dataset_name": dataset_name,
            "shape": df.shape,
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024**2), 2),
            "columns": list(df.columns),
            "dtypes": df.dtypes.value_counts().to_dict(),
        }

        # Statistiques de complétude
        analysis["completeness"] = {
            "total_cells": df.size,
            "non_null_cells": df.notna().sum().sum(),
            "null_cells": df.isna().sum().sum(),
            "completeness_rate": round(df.notna().sum().sum() / df.size * 100, 2),
        }

        # Statistiques par colonne
        column_stats = []
        for col in df.columns:
            stat = {
                "column": col,
                "dtype": str(df[col].dtype),
                "non_null": int(df[col].notna().sum()),
                "null": int(df[col].isna().sum()),
                "null_pct": round(df[col].isna().sum() / len(df) * 100, 2),
                "unique": int(df[col].nunique()),
            }

            if pd.api.types.is_numeric_dtype(df[col]):
                stat.update(
                    {
                        "min": df[col].min(),
                        "max": df[col].max(),
                        "mean": round(df[col].mean(), 3),
                        "median": df[col].median(),
                        "std": round(df[col].std(), 3),
                    }
                )

            column_stats.append(stat)

        analysis["column_statistics"] = pd.DataFrame(column_stats)

        return analysis

    def identify_temporal_variables(self, df: pd.DataFrame) -> List[str]:
        """Identifie les variables temporelles"""
        temporal_vars = []

        for col in df.columns:
            if any(
                keyword in col.lower() for keyword in ["year", "année", "date", "time"]
            ):
                temporal_vars.append(col)
            elif df[col].dtype == "datetime64[ns]":
                temporal_vars.append(col)

        return temporal_vars

    def identify_spatial_variables(self, df: pd.DataFrame) -> List[str]:
        """Identifie les variables spatiales"""
        spatial_vars = []

        spatial_keywords = [
            "région",
            "region",
            "département",
            "department",
            "commune",
            "ville",
            "city",
            "localité",
            "locality",
            "latitude",
            "longitude",
            "admin",
        ]

        for col in df.columns:
            if any(keyword in col.lower() for keyword in spatial_keywords):
                spatial_vars.append(col)

        return spatial_vars

    def generate_summary_report(
        self, analyses: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """Génère un rapport de synthèse"""
        summary_data = []

        for dataset_name, analysis in analyses.items():
            summary_data.append(
                {
                    "dataset": dataset_name,
                    "rows": analysis["shape"][0],
                    "columns": analysis["shape"][1],
                    "memory_mb": analysis["memory_usage_mb"],
                    "completeness_rate": analysis["completeness"]["completeness_rate"],
                    "null_cells": analysis["completeness"]["null_cells"],
                }
            )

        return pd.DataFrame(summary_data)


# %%
class TrendAnalyzer:
    """Analyseur de tendances temporelles"""

    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    def analyze_temporal_trends(
        self, df: pd.DataFrame, time_col: str, value_cols: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """Analyse les tendances temporelles"""
        self.logger.info(f"📈 Analyse des tendances temporelles")

        trends = {}

        for value_col in value_cols:
            if value_col not in df.columns or not pd.api.types.is_numeric_dtype(
                df[value_col]
            ):
                continue

            try:
                # Agrégation par période temporelle
                trend_df = (
                    df.groupby(time_col)[value_col]
                    .agg(["count", "mean", "median", "std", "min", "max"])
                    .reset_index()
                )

                # Calcul des variations
                trend_df["pct_change"] = trend_df["mean"].pct_change() * 100
                trend_df["cumulative_change"] = (
                    (trend_df["mean"] / trend_df["mean"].iloc[0]) - 1
                ) * 100

                trends[value_col] = trend_df

            except Exception as e:
                self.logger.warning(f"⚠️ Erreur tendance {value_col}: {e}")

        return trends

    def calculate_growth_rates(
        self, df: pd.DataFrame, time_col: str, value_col: str
    ) -> pd.DataFrame:
        """Calcule les taux de croissance"""
        result = df.copy()
        result = result.sort_values(time_col)

        # Taux de croissance annuel
        result["growth_rate"] = result[value_col].pct_change() * 100

        # Taux de croissance annuel moyen (CAGR)
        if len(result) > 1:
            first_value = result[value_col].iloc[0]
            last_value = result[value_col].iloc[-1]
            n_periods = len(result) - 1

            if first_value > 0 and last_value > 0:
                cagr = (((last_value / first_value) ** (1 / n_periods)) - 1) * 100
                result["cagr"] = cagr

        return result


# %%
class SpatialAnalyzer:
    """Analyseur de dynamiques spatiales"""

    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    def analyze_spatial_distribution(
        self, df: pd.DataFrame, spatial_col: str, value_cols: List[str]
    ) -> pd.DataFrame:
        """Analyse la distribution spatiale"""
        self.logger.info(f"🗺️ Analyse spatiale sur {spatial_col}")

        spatial_stats = []

        for value_col in value_cols:
            if value_col not in df.columns or not pd.api.types.is_numeric_dtype(
                df[value_col]
            ):
                continue

            try:
                stats_by_location = (
                    df.groupby(spatial_col)[value_col]
                    .agg(["count", "mean", "median", "std", "min", "max", "sum"])
                    .reset_index()
                )

                stats_by_location["variable"] = value_col
                stats_by_location["cv"] = (
                    stats_by_location["std"] / stats_by_location["mean"]
                ) * 100

                spatial_stats.append(stats_by_location)

            except Exception as e:
                self.logger.warning(f"⚠️ Erreur spatiale {value_col}: {e}")

        if spatial_stats:
            return pd.concat(spatial_stats, ignore_index=True)

        return pd.DataFrame()

    def calculate_regional_disparities(
        self, df: pd.DataFrame, spatial_col: str, value_col: str
    ) -> Dict[str, float]:
        """Calcule les disparités régionales"""
        regional_means = df.groupby(spatial_col)[value_col].mean()

        disparities = {
            "gini_coefficient": self._calculate_gini(regional_means.values),
            "coefficient_variation": (regional_means.std() / regional_means.mean())
            * 100,
            "range_ratio": (
                regional_means.max() / regional_means.min()
                if regional_means.min() > 0
                else np.nan
            ),
            "max_value": regional_means.max(),
            "min_value": regional_means.min(),
            "mean_value": regional_means.mean(),
        }

        return disparities

    def _calculate_gini(self, values: np.ndarray) -> float:
        """Calcule le coefficient de Gini"""
        values = np.array(values)
        values = values[~np.isnan(values)]

        if len(values) == 0:
            return np.nan

        values = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)

        return (2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n


# %%
class CorrelationAnalyzer:
    """Analyseur de corrélations"""

    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    def calculate_correlations(
        self, df: pd.DataFrame, method: str = "pearson"
    ) -> pd.DataFrame:
        """Calcule la matrice de corrélation"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            self.logger.warning(
                "⚠️ Pas assez de colonnes numériques pour la corrélation"
            )
            return pd.DataFrame()

        if method == "pearson":
            corr_matrix = df[numeric_cols].corr(method="pearson")
        elif method == "spearman":
            corr_matrix = df[numeric_cols].corr(method="spearman")
        else:
            corr_matrix = df[numeric_cols].corr()

        return corr_matrix

    def find_strong_correlations(
        self, corr_matrix: pd.DataFrame, threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """Identifie les corrélations fortes"""
        threshold = threshold or self.config.CORRELATION_THRESHOLD

        strong_corr = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]

                if abs(corr_value) >= threshold:
                    strong_corr.append(
                        {
                            "variable_1": var1,
                            "variable_2": var2,
                            "correlation": round(corr_value, 3),
                            "strength": (
                                "forte" if abs(corr_value) >= 0.8 else "modérée"
                            ),
                        }
                    )

        if not strong_corr:
            return pd.DataFrame(
                columns=["variable_1", "variable_2", "correlation", "strength"]
            )

        return pd.DataFrame(strong_corr).sort_values(
            "correlation", key=abs, ascending=False
        )

    def cross_dataset_correlation(
        self, df1: pd.DataFrame, df2: pd.DataFrame, merge_cols: List[str]
    ) -> pd.DataFrame:
        """Corrélations croisées entre datasets"""
        try:
            merged = pd.merge(
                df1, df2, on=merge_cols, how="inner", suffixes=("_1", "_2")
            )

            numeric_cols = merged.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) >= 2:
                return self.calculate_correlations(merged[numeric_cols])

        except Exception as e:
            self.logger.error(f"❌ Erreur corrélation croisée: {e}")

        return pd.DataFrame()


# %%
class AnomalyDetector:
    """Détecteur d'anomalies"""

    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.anomaly_reports: List[AnomalyReport] = []

    def detect_zscore_anomalies(
        self, df: pd.DataFrame, dataset_name: str
    ) -> pd.DataFrame:
        """Détecte les anomalies par Z-score"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        anomalies_df = pd.DataFrame()

        for col in numeric_cols:
            if df[col].notna().sum() < 3:
                continue

            z_scores = np.abs(zscore(df[col].dropna()))
            anomaly_mask = z_scores > self.config.ZSCORE_THRESHOLD

            if anomaly_mask.any():
                anomaly_count = anomaly_mask.sum()
                anomaly_pct = (anomaly_count / len(df)) * 100

                report = AnomalyReport(
                    dataset_name=dataset_name,
                    variable=col,
                    anomaly_type="zscore",
                    anomaly_count=anomaly_count,
                    anomaly_percentage=anomaly_pct,
                    details={"threshold": self.config.ZSCORE_THRESHOLD},
                )
                self.anomaly_reports.append(report)

        return anomalies_df

    def detect_iqr_anomalies(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Détecte les anomalies par IQR"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        anomalies_list = []

        for col in numeric_cols:
            if df[col].notna().sum() < 4:
                continue

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - self.config.IQR_MULTIPLIER * IQR
            upper_bound = Q3 + self.config.IQR_MULTIPLIER * IQR

            anomaly_mask = (df[col] < lower_bound) | (df[col] > upper_bound)

            if anomaly_mask.any():
                anomaly_count = anomaly_mask.sum()
                anomaly_pct = (anomaly_count / len(df)) * 100

                report = AnomalyReport(
                    dataset_name=dataset_name,
                    variable=col,
                    anomaly_type="iqr",
                    anomaly_count=anomaly_count,
                    anomaly_percentage=anomaly_pct,
                    details={
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound,
                        "Q1": Q1,
                        "Q3": Q3,
                        "IQR": IQR,
                    },
                )
                self.anomaly_reports.append(report)

                anomalies_list.append(
                    {
                        "dataset": dataset_name,
                        "variable": col,
                        "anomaly_indices": df[anomaly_mask].index.tolist(),
                    }
                )

        return pd.DataFrame(anomalies_list) if anomalies_list else pd.DataFrame()

    def detect_inconsistencies(
        self, df: pd.DataFrame, dataset_name: str
    ) -> List[Dict[str, Any]]:
        """Détecte les incohérences"""
        inconsistencies = []

        # Vérification des valeurs négatives inappropriées
        for col in df.select_dtypes(include=[np.number]).columns:
            if any(
                keyword in col.lower()
                for keyword in ["population", "count", "nombre", "effectif"]
            ):
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    inconsistencies.append(
                        {
                            "dataset": dataset_name,
                            "variable": col,
                            "issue": "valeurs_négatives",
                            "count": negative_count,
                            "description": f"{negative_count} valeurs négatives pour une variable de comptage",
                        }
                    )

        # Vérification des années invalides
        for col in df.columns:
            if "year" in col.lower() or "année" in col.lower():
                invalid_years = df[(df[col] < 1900) | (df[col] > 2025)]
                if len(invalid_years) > 0:
                    inconsistencies.append(
                        {
                            "dataset": dataset_name,
                            "variable": col,
                            "issue": "années_invalides",
                            "count": len(invalid_years),
                            "description": f"{len(invalid_years)} années hors de la plage [1900-2025]",
                        }
                    )

        return inconsistencies

    def generate_anomaly_report(self) -> pd.DataFrame:
        """Génère le rapport d'anomalies"""
        if not self.anomaly_reports:
            return pd.DataFrame()

        return pd.DataFrame([report.to_dict() for report in self.anomaly_reports])


# %%
class IndicatorBuilder:
    """Constructeur d'indicateurs dérivés"""

    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    def build_demographic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construit les indicateurs démographiques"""
        result = df.copy()

        # Taux de croissance de la population
        if "population" in df.columns or "total_population" in df.columns:
            pop_col = "population" if "population" in df.columns else "total_population"

            if "year" in df.columns or "année" in df.columns:
                time_col = "year" if "year" in df.columns else "année"
                result = result.sort_values(time_col)
                result["population_growth_rate"] = result[pop_col].pct_change() * 100

        # Ratio de population jeune
        if "population_0_14" in df.columns and "total_population" in df.columns:
            result["youth_population_ratio"] = (
                result["population_0_14"] / result["total_population"]
            ) * 100

        # Ratio de population âgée
        if "population_65_plus" in df.columns and "total_population" in df.columns:
            result["elderly_population_ratio"] = (
                result["population_65_plus"] / result["total_population"]
            ) * 100

        # Densité de population (si superficie disponible)
        if all(col in df.columns for col in ["total_population", "surface_area_km2"]):
            result["population_density"] = (
                result["total_population"] / result["surface_area_km2"]
            )

        self.logger.info("✅ Indicateurs démographiques créés")
        return result

    def build_economic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construit les indicateurs économiques"""
        result = df.copy()

        # PIB par habitant
        if all(col in df.columns for col in ["gdp", "total_population"]):
            result["gdp_per_capita"] = result["gdp"] / result["total_population"]

        # Taux de croissance du PIB
        if "gdp" in df.columns:
            if "year" in df.columns or "année" in df.columns:
                time_col = "year" if "year" in df.columns else "année"
                result = result.sort_values(time_col)
                result["gdp_growth_rate"] = result["gdp"].pct_change() * 100

        # Indice de PIB (base 100)
        if "gdp" in df.columns and "year" in df.columns:
            base_year = self.config.GDP_BASE_YEAR
            base_gdp = (
                result[result["year"] == base_year]["gdp"].iloc[0]
                if base_year in result["year"].values
                else result["gdp"].iloc[0]
            )
            result["gdp_index"] = (result["gdp"] / base_gdp) * 100

        self.logger.info("✅ Indicateurs économiques créés")
        return result

    def build_education_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construit les indicateurs d'éducation"""
        result = df.copy()

        # Taux de scolarisation net
        if all(
            col in df.columns for col in ["enrolled_students", "school_age_population"]
        ):
            result["net_enrollment_rate"] = (
                result["enrolled_students"] / result["school_age_population"]
            ) * 100

        # Ratio élèves-enseignants
        if all(col in df.columns for col in ["total_students", "total_teachers"]):
            result["student_teacher_ratio"] = (
                result["total_students"] / result["total_teachers"]
            )

        self.logger.info("✅ Indicateurs d'éducation créés")
        return result

    def build_composite_index(
        self,
        df: pd.DataFrame,
        indicators: List[str],
        weights: Optional[List[float]] = None,
        index_name: str = "composite_index",
    ) -> pd.DataFrame:
        """Construit un indice composite"""
        result = df.copy()

        available_indicators = [ind for ind in indicators if ind in df.columns]

        if not available_indicators:
            self.logger.warning(f"⚠️ Aucun indicateur disponible pour {index_name}")
            return result

        # Normalisation des indicateurs (min-max)
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(df[available_indicators].fillna(0))
        normalized_df = pd.DataFrame(
            normalized_data, columns=available_indicators, index=df.index
        )

        # Application des poids
        if weights is None:
            weights = [1 / len(available_indicators)] * len(available_indicators)

        # Calcul de l'indice composite
        result[index_name] = sum(
            normalized_df[ind] * weight
            for ind, weight in zip(available_indicators, weights)
        )

        # Normalisation finale (0-100)
        result[index_name] = (
            (result[index_name] - result[index_name].min())
            / (result[index_name].max() - result[index_name].min())
        ) * 100

        self.logger.info(f"✅ Indice composite '{index_name}' créé")
        return result

    def build_regional_development_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construit un indice de développement régional"""
        indicators = []

        # Sélection automatique des indicateurs disponibles
        if "gdp_per_capita" in df.columns:
            indicators.append("gdp_per_capita")
        if "net_enrollment_rate" in df.columns:
            indicators.append("net_enrollment_rate")
        if "life_expectancy" in df.columns:
            indicators.append("life_expectancy")
        if "access_electricity" in df.columns:
            indicators.append("access_electricity")

        if indicators:
            return self.build_composite_index(
                df, indicators, index_name="regional_development_index"
            )

        self.logger.warning("⚠️ Pas assez d'indicateurs pour l'indice de développement")
        return df


# %%
class AggregationEngine:
    """Moteur d'agrégation temporelle et spatiale"""

    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    def temporal_aggregation(
        self,
        df: pd.DataFrame,
        time_col: str,
        value_cols: List[str],
        agg_functions: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        if agg_functions is None:
            agg_functions = {col: "mean" for col in value_cols}

        valid_agg = {
            col: func
            for col, func in agg_functions.items()
            if col in df.columns and col != time_col
        }

        if not valid_agg:
            self.logger.warning("⚠️ Aucune colonne valide pour l'agrégation")
            return pd.DataFrame()

        aggregated = df.groupby(time_col).agg(valid_agg).reset_index()

        aggregated.columns = [
            f"{col}_{func}" if col != time_col else col
            for col, func in zip(
                aggregated.columns, [time_col] + list(valid_agg.values())
            )
        ]

        self.logger.info(f"✅ Agrégation temporelle: {len(aggregated)} périodes")
        return aggregated

    def spatial_aggregation(
        self,
        df: pd.DataFrame,
        spatial_col: str,
        value_cols: List[str],
        agg_functions: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """Agrégation spatiale"""
        if agg_functions is None:
            agg_functions = {col: "sum" for col in value_cols}

        valid_agg = {
            col: func
            for col, func in agg_functions.items()
            if col in df.columns and col != spatial_col
        }

        if not valid_agg:
            self.logger.warning("⚠️ Aucune colonne valide pour l'agrégation spatiale")
            return pd.DataFrame()

        aggregated = df.groupby(spatial_col).agg(valid_agg).reset_index()

        self.logger.info(f"✅ Agrégation spatiale: {len(aggregated)} zones")
        return aggregated

    def normalize_by_population(
        self,
        df: pd.DataFrame,
        value_cols: List[str],
        population_col: str = "total_population",
    ) -> pd.DataFrame:
        """Normalise les valeurs par habitant"""
        result = df.copy()

        if population_col not in df.columns:
            self.logger.warning(f"⚠️ Colonne {population_col} introuvable")
            return result

        for col in value_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                new_col_name = f"{col}_per_capita"
                result[new_col_name] = result[col] / result[population_col]
                self.logger.info(f"✅ Créé: {new_col_name}")

        return result

    def multi_level_aggregation(
        self, df: pd.DataFrame, group_cols: List[str], value_cols: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """Agrégation multi-niveaux"""
        aggregations = {}

        for i in range(1, len(group_cols) + 1):
            level_cols = group_cols[:i]
            level_name = "_".join(level_cols)

            agg_dict = {
                col: ["sum", "mean", "count"] for col in value_cols if col in df.columns
            }

            if agg_dict:
                agg_result = df.groupby(level_cols).agg(agg_dict).reset_index()
                agg_result.columns = [
                    "_".join(col).strip("_") for col in agg_result.columns
                ]
                aggregations[level_name] = agg_result
                self.logger.info(f"✅ Agrégation niveau {i}: {len(agg_result)} groupes")

        return aggregations


# %%
class VisualizationEngine:
    """Moteur de visualisation"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.logger = logging.getLogger(self.__class__.__name__)

    def plot_temporal_trends(
        self, trends: Dict[str, pd.DataFrame], time_col: str, save: bool = True
    ) -> None:
        """Visualise les tendances temporelles"""
        n_plots = len(trends)
        if n_plots == 0:
            return

        fig, axes = plt.subplots(min(n_plots, 3), 1, figsize=(14, 4 * min(n_plots, 3)))
        if n_plots == 1:
            axes = [axes]

        for idx, (var_name, trend_df) in enumerate(list(trends.items())[:3]):
            ax = axes[idx]
            ax.plot(
                trend_df[time_col],
                trend_df["mean"],
                marker="o",
                linewidth=2,
                label="Moyenne",
            )
            ax.fill_between(
                trend_df[time_col],
                trend_df["mean"] - trend_df["std"],
                trend_df["mean"] + trend_df["std"],
                alpha=0.3,
                label="±1 écart-type",
            )
            ax.set_title(
                f"Évolution temporelle: {var_name}", fontsize=12, fontweight="bold"
            )
            ax.set_xlabel("Année")
            ax.set_ylabel("Valeur")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / "temporal_trends.png"
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            self.logger.info(f"💾 Graphique sauvegardé: {filepath.name}")

        plt.close()

    def plot_correlation_heatmap(
        self, corr_matrix: pd.DataFrame, save: bool = True
    ) -> None:
        """Visualise la matrice de corrélation"""
        if corr_matrix.empty:
            return

        fig, ax = plt.subplots(figsize=(12, 10))

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )

        ax.set_title("Matrice de corrélation", fontsize=14, fontweight="bold", pad=20)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / "correlation_heatmap.png"
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            self.logger.info(f"💾 Heatmap sauvegardée: {filepath.name}")

        plt.close()

    def plot_spatial_distribution(
        self,
        spatial_stats: pd.DataFrame,
        spatial_col: str,
        value_col: str = "mean",
        save: bool = True,
    ) -> None:
        """Visualise la distribution spatiale"""
        if spatial_stats.empty:
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        spatial_stats_sorted = spatial_stats.sort_values(
            value_col, ascending=False
        ).head(20)

        bars = ax.barh(
            spatial_stats_sorted[spatial_col], spatial_stats_sorted[value_col]
        )

        # Gradient de couleurs
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_xlabel("Valeur moyenne")
        ax.set_title(f"Distribution spatiale (Top 20)", fontsize=14, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / "spatial_distribution.png"
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            self.logger.info(f"💾 Distribution sauvegardée: {filepath.name}")

        plt.close()


# %%
class ExplorationOrchestrator:
    """Orchestrateur de l'exploration et analyse"""

    def __init__(
        self, config: Optional[AnalysisConfig] = None, base_dir: Optional[Path] = None
    ):
        self.config = config or AnalysisConfig()
        self.base_dir = base_dir or Path(".")
        self.logger = logging.getLogger(__name__)

        # Initialisation des composants
        self.dir_manager = DirectoryManager(self.base_dir, self.config)
        self.directories = self.dir_manager.initialize_structure()

        self.loader = DataLoader(self.directories["input"])
        self.descriptive_analyzer = DescriptiveAnalyzer(self.config)
        self.trend_analyzer = TrendAnalyzer(self.config)
        self.spatial_analyzer = SpatialAnalyzer(self.config)
        self.correlation_analyzer = CorrelationAnalyzer(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        self.indicator_builder = IndicatorBuilder(self.config)
        self.aggregation_engine = AggregationEngine(self.config)
        self.viz_engine = VisualizationEngine(self.directories["visualizations"])

        self.results = {}

    def run_descriptive_analysis(
        self, datasets: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Phase 1: Analyse descriptive"""
        self.logger.info("📊 PHASE 1: ANALYSE DESCRIPTIVE")
        print("\n" + "=" * 80)
        print("📊 PHASE 1: ANALYSE DESCRIPTIVE")
        print("=" * 80)

        analyses = {}

        for name, df in datasets.items():
            self.logger.info(f"▶️ Analyse: {name}")
            analysis = self.descriptive_analyzer.analyze_dataset(df, name)
            analyses[name] = analysis

            # Sauvegarde des statistiques
            if "column_statistics" in analysis:
                stats_path = self.directories["analysis"] / f"{name}_column_stats.csv"
                analysis["column_statistics"].to_csv(stats_path, index=False)

        # Rapport de synthèse
        summary_report = self.descriptive_analyzer.generate_summary_report(analyses)
        summary_path = self.directories["analysis"] / "descriptive_summary.csv"
        summary_report.to_csv(summary_path, index=False)

        print("\n✅ Analyse descriptive terminée")
        print(f"   Datasets analysés: {len(analyses)}")
        print(f"   Rapport: {summary_path.name}")

        return analyses

    def run_trend_analysis(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Phase 2: Analyse des tendances"""
        self.logger.info("📈 PHASE 2: ANALYSE DES TENDANCES TEMPORELLES")
        print("\n" + "=" * 80)
        print("📈 PHASE 2: ANALYSE DES TENDANCES TEMPORELLES")
        print("=" * 80)

        all_trends = {}

        for name, df in datasets.items():
            temporal_vars = self.descriptive_analyzer.identify_temporal_variables(df)

            if not temporal_vars:
                continue

            time_col = temporal_vars[0]
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            value_cols = [col for col in numeric_cols if col != time_col][:5]

            if value_cols:
                trends = self.trend_analyzer.analyze_temporal_trends(
                    df, time_col, value_cols
                )

                if trends:
                    all_trends[name] = trends

                    # Sauvegarde
                    for var, trend_df in trends.items():
                        trend_path = (
                            self.directories["analysis"] / f"{name}_{var}_trend.csv"
                        )
                        trend_df.to_csv(trend_path, index=False)

                    # Visualisation
                    self.viz_engine.plot_temporal_trends(trends, time_col)

        print(f"\n✅ Analyse des tendances terminée")
        print(f"   Datasets avec tendances: {len(all_trends)}")

        return all_trends

    def run_spatial_analysis(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Phase 3: Analyse spatiale"""
        self.logger.info("🗺️ PHASE 3: ANALYSE DES DYNAMIQUES SPATIALES")
        print("\n" + "=" * 80)
        print("🗺️ PHASE 3: ANALYSE DES DYNAMIQUES SPATIALES")
        print("=" * 80)

        spatial_results = {}

        for name, df in datasets.items():
            spatial_vars = self.descriptive_analyzer.identify_spatial_variables(df)

            if not spatial_vars:
                continue

            spatial_col = spatial_vars[0]
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            value_cols = [col for col in numeric_cols if col != spatial_col][:5]

            if value_cols:
                spatial_dist = self.spatial_analyzer.analyze_spatial_distribution(
                    df, spatial_col, value_cols
                )

                if not spatial_dist.empty:
                    spatial_results[name] = spatial_dist

                    # Sauvegarde
                    spatial_path = (
                        self.directories["analysis"] / f"{name}_spatial_analysis.csv"
                    )
                    spatial_dist.to_csv(spatial_path, index=False)

                    # Visualisation
                    self.viz_engine.plot_spatial_distribution(spatial_dist, spatial_col)

        print(f"\n✅ Analyse spatiale terminée")
        print(f"   Datasets analysés spatialement: {len(spatial_results)}")

        return spatial_results

    def run_correlation_analysis(
        self, datasets: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Phase 4: Analyse des corrélations"""
        self.logger.info("🔗 PHASE 4: ANALYSE DES CORRÉLATIONS")
        print("\n" + "=" * 80)
        print("🔗 PHASE 4: ANALYSE DES CORRÉLATIONS")
        print("=" * 80)

        correlations = {}

        for name, df in datasets.items():
            corr_matrix = self.correlation_analyzer.calculate_correlations(df)

            if not corr_matrix.empty:
                correlations[name] = {
                    "matrix": corr_matrix,
                    "strong_correlations": self.correlation_analyzer.find_strong_correlations(
                        corr_matrix
                    ),
                }

                # Sauvegarde
                corr_path = self.directories["analysis"] / f"{name}_correlations.csv"
                corr_matrix.to_csv(corr_path)

                strong_corr_path = (
                    self.directories["analysis"] / f"{name}_strong_correlations.csv"
                )
                if not correlations[name]["strong_correlations"].empty:
                    correlations[name]["strong_correlations"].to_csv(
                        strong_corr_path, index=False
                    )

                # Visualisation
                self.viz_engine.plot_correlation_heatmap(corr_matrix)

        print(f"\n✅ Analyse des corrélations terminée")
        print(f"   Datasets analysés: {len(correlations)}")

        return correlations

    def run_anomaly_detection(
        self, datasets: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Phase 5: Détection des anomalies"""
        self.logger.info("🔍 PHASE 5: DÉTECTION DES ANOMALIES")
        print("\n" + "=" * 80)
        print("🔍 PHASE 5: DÉTECTION DES ANOMALIES")
        print("=" * 80)

        all_anomalies = {}
        all_inconsistencies = []

        for name, df in datasets.items():
            # Détection Z-score
            self.anomaly_detector.detect_zscore_anomalies(df, name)

            # Détection IQR
            iqr_anomalies = self.anomaly_detector.detect_iqr_anomalies(df, name)

            # Détection d'incohérences
            inconsistencies = self.anomaly_detector.detect_inconsistencies(df, name)

            if not iqr_anomalies.empty:
                all_anomalies[name] = iqr_anomalies

            if inconsistencies:
                all_inconsistencies.extend(inconsistencies)

        # Rapport global
        anomaly_report = self.anomaly_detector.generate_anomaly_report()
        if not anomaly_report.empty:
            anomaly_path = self.directories["anomalies"] / "anomaly_report.csv"
            anomaly_report.to_csv(anomaly_path, index=False)
            print(f"\n📄 Rapport d'anomalies: {anomaly_path.name}")
            print(f"   Total anomalies détectées: {len(anomaly_report)}")

        # Rapport d'incohérences
        if all_inconsistencies:
            incon_df = pd.DataFrame(all_inconsistencies)
            incon_path = self.directories["anomalies"] / "inconsistencies_report.csv"
            incon_df.to_csv(incon_path, index=False)
            print(f"📄 Rapport d'incohérences: {incon_path.name}")
            print(f"   Total incohérences: {len(all_inconsistencies)}")

        print(f"\n✅ Détection des anomalies terminée")

        return {
            "anomalies": all_anomalies,
            "anomaly_report": anomaly_report,
            "inconsistencies": all_inconsistencies,
        }

    def run_indicator_creation(
        self, datasets: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Phase 6: Création d'indicateurs"""
        self.logger.info("🔧 PHASE 6: CRÉATION D'INDICATEURS DÉRIVÉS")
        print("\n" + "=" * 80)
        print("🔧 PHASE 6: CRÉATION D'INDICATEURS DÉRIVÉS")
        print("=" * 80)

        enriched_datasets = {}

        for name, df in datasets.items():
            enriched = df.copy()

            # Indicateurs démographiques
            enriched = self.indicator_builder.build_demographic_indicators(enriched)

            # Indicateurs économiques
            enriched = self.indicator_builder.build_economic_indicators(enriched)

            # Indicateurs d'éducation
            enriched = self.indicator_builder.build_education_indicators(enriched)

            # Indice de développement régional
            enriched = self.indicator_builder.build_regional_development_index(enriched)

            # Comptage des nouvelles colonnes
            new_cols = set(enriched.columns) - set(df.columns)

            if new_cols:
                enriched_datasets[name] = enriched

                # Sauvegarde
                enriched_path = self.directories["enriched"] / f"{name}_enriched.csv"
                enriched.to_csv(enriched_path, index=False)

                print(f"\n✅ {name}: {len(new_cols)} nouveaux indicateurs")
                for col in sorted(new_cols):
                    print(f"   - {col}")

        print(f"\n✅ Création d'indicateurs terminée")
        print(f"   Datasets enrichis: {len(enriched_datasets)}")

        return enriched_datasets

    def run_aggregations(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Phase 7: Agrégations"""
        self.logger.info("📊 PHASE 7: AGRÉGATIONS TEMPORELLES ET SPATIALES")
        print("\n" + "=" * 80)
        print("📊 PHASE 7: AGRÉGATIONS TEMPORELLES ET SPATIALES")
        print("=" * 80)

        aggregations = {}

        for name, df in datasets.items():
            dataset_aggs = {}

            # Agrégation temporelle
            temporal_vars = self.descriptive_analyzer.identify_temporal_variables(df)
            if temporal_vars:
                time_col = temporal_vars[0]
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                value_cols = [col for col in numeric_cols if col != time_col][:10]

                if value_cols:
                    temp_agg = self.aggregation_engine.temporal_aggregation(
                        df, time_col, value_cols
                    )
                    if not temp_agg.empty:
                        dataset_aggs["temporal"] = temp_agg
                        temp_path = (
                            self.directories["processed"] / f"{name}_temporal_agg.csv"
                        )
                        temp_agg.to_csv(temp_path, index=False)

            # Agrégation spatiale
            spatial_vars = self.descriptive_analyzer.identify_spatial_variables(df)
            if spatial_vars:
                spatial_col = spatial_vars[0]
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                value_cols = [col for col in numeric_cols][:10]

                if value_cols:
                    spatial_agg = self.aggregation_engine.spatial_aggregation(
                        df, spatial_col, value_cols
                    )
                    if not spatial_agg.empty:
                        dataset_aggs["spatial"] = spatial_agg
                        spatial_path = (
                            self.directories["processed"] / f"{name}_spatial_agg.csv"
                        )
                        spatial_agg.to_csv(spatial_path, index=False)

            # Normalisation par population
            if "total_population" in df.columns or "population" in df.columns:
                pop_col = (
                    "total_population"
                    if "total_population" in df.columns
                    else "population"
                )
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                value_cols = [col for col in numeric_cols if col != pop_col][:5]

                if value_cols:
                    normalized = self.aggregation_engine.normalize_by_population(
                        df, value_cols, pop_col
                    )
                    new_cols = [
                        col for col in normalized.columns if "_per_capita" in col
                    ]

                    if new_cols:
                        dataset_aggs["per_capita"] = normalized[new_cols + [pop_col]]
                        norm_path = (
                            self.directories["processed"] / f"{name}_per_capita.csv"
                        )
                        normalized.to_csv(norm_path, index=False)

            if dataset_aggs:
                aggregations[name] = dataset_aggs
                print(f"\n✅ {name}: {len(dataset_aggs)} types d'agrégation")

        print(f"\n✅ Agrégations terminées")
        print(f"   Datasets agrégés: {len(aggregations)}")

        return aggregations

    def generate_methodology_document(self) -> pd.DataFrame:
        """Génère la documentation méthodologique"""
        self.logger.info("📝 Génération de la documentation méthodologique")

        methodology = []

        # Anomalies
        anomaly_report = self.anomaly_detector.generate_anomaly_report()
        if not anomaly_report.empty:
            for _, row in anomaly_report.iterrows():
                methodology.append(
                    {
                        "category": "Anomalie",
                        "dataset": row["dataset"],
                        "variable": row["variable"],
                        "type": row["anomaly_type"],
                        "action": f"{row['count']} valeurs détectées ({row['percentage']}%)",
                        "justification": f"Seuil: {self.config.ZSCORE_THRESHOLD if row['anomaly_type'] == 'zscore' else self.config.IQR_MULTIPLIER}",
                    }
                )

        # Indicateurs créés
        methodology.append(
            {
                "category": "Indicateur",
                "dataset": "Général",
                "variable": "population_growth_rate",
                "type": "Dérivé",
                "action": "Taux de croissance calculé",
                "justification": "Variation annuelle en pourcentage",
            }
        )

        methodology.append(
            {
                "category": "Indicateur",
                "dataset": "Général",
                "variable": "gdp_per_capita",
                "type": "Ratio",
                "action": "PIB / Population",
                "justification": "Normalisation économique",
            }
        )

        methodology.append(
            {
                "category": "Indicateur",
                "dataset": "Général",
                "variable": "regional_development_index",
                "type": "Composite",
                "action": "Indice multi-dimensionnel",
                "justification": "Combinaison normalisée de plusieurs indicateurs",
            }
        )

        # Agrégations
        methodology.append(
            {
                "category": "Agrégation",
                "dataset": "Général",
                "variable": "temporal_aggregation",
                "type": "Temporelle",
                "action": "Agrégation par année",
                "justification": "Analyse des tendances historiques",
            }
        )

        methodology.append(
            {
                "category": "Agrégation",
                "dataset": "Général",
                "variable": "spatial_aggregation",
                "type": "Spatiale",
                "action": "Agrégation par région",
                "justification": "Comparaisons géographiques",
            }
        )

        methodology_df = pd.DataFrame(methodology)

        # Sauvegarde
        method_path = self.directories["analysis"] / "methodology_documentation.csv"
        methodology_df.to_csv(method_path, index=False)

        self.logger.info(f"✅ Documentation: {method_path.name}")

        return methodology_df

    def run_complete_analysis(self) -> Dict[str, Any]:
        """Exécute le pipeline complet d'analyse"""
        print("\n" + "=" * 80)
        print("🚀 DÉMARRAGE PIPELINE ANALYSE COMPLÈTE - TÂCHE 2")
        print("=" * 80)

        # Chargement des données
        datasets = self.loader.load_all_datasets()

        if not datasets:
            self.logger.error(
                "❌ Aucune donnée chargée. Vérifiez le répertoire d'entrée."
            )
            return {}

        # Phase 1: Analyse descriptive
        descriptive_results = self.run_descriptive_analysis(datasets)

        # Phase 2: Tendances temporelles
        trend_results = self.run_trend_analysis(datasets)

        # Phase 3: Dynamiques spatiales
        spatial_results = self.run_spatial_analysis(datasets)

        # Phase 4: Corrélations
        correlation_results = self.run_correlation_analysis(datasets)

        # Phase 5: Anomalies
        anomaly_results = self.run_anomaly_detection(datasets)

        # Phase 6: Indicateurs dérivés
        enriched_datasets = self.run_indicator_creation(datasets)

        # Phase 7: Agrégations
        aggregation_results = self.run_aggregations(
            enriched_datasets if enriched_datasets else datasets
        )

        # Documentation méthodologique
        methodology_doc = self.generate_methodology_document()

        # Rapport final
        print("\n" + "=" * 80)
        print("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
        print("=" * 80)
        print(f"\n📊 RÉSULTATS:")
        print(f"   - Datasets analysés: {len(datasets)}")
        print(f"   - Datasets enrichis: {len(enriched_datasets)}")
        print(f"   - Tendances identifiées: {len(trend_results)}")
        print(f"   - Analyses spatiales: {len(spatial_results)}")
        print(f"   - Matrices de corrélation: {len(correlation_results)}")
        print(
            f"   - Anomalies détectées: {len(anomaly_results.get('anomaly_report', []))}"
        )
        print(f"\n📂 LIVRABLES:")
        print(f"   - Données enrichies: {self.directories['enriched']}")
        print(f"   - Analyses: {self.directories['analysis']}")
        print(f"   - Rapports d'anomalies: {self.directories['anomalies']}")
        print(f"   - Visualisations: {self.directories['visualizations']}")
        print(f"   - Données agrégées: {self.directories['processed']}")
        print("=" * 80 + "\n")

        return {
            "datasets": datasets,
            "descriptive_analysis": descriptive_results,
            "trends": trend_results,
            "spatial_analysis": spatial_results,
            "correlations": correlation_results,
            "anomalies": anomaly_results,
            "enriched_datasets": enriched_datasets,
            "aggregations": aggregation_results,
            "methodology": methodology_doc,
        }


# %%
def main():
    """Point d'entrée principal"""

    # Configuration de l'environnement
    setup_analysis_environment(log_dir=Path("logs_task_2"))

    # Création de l'orchestrateur
    orchestrator = ExplorationOrchestrator()

    # Exécution du pipeline complet
    results = orchestrator.run_complete_analysis()

    return results


# %%
if __name__ == "__main__":
    results = main()
