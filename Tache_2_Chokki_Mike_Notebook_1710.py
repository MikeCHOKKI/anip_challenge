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


# %%
@dataclass
class AnalysisConfig:
    """
    Configuration class for analysis tasks.

    Provides configurations such as country details, anomaly detection
    thresholds, indicator configurations, and directory structure
    required for data analysis tasks.

    Attributes:
        COUNTRY_CODE: ISO 3166-1 alpha-2 country code.
        COUNTRY_NAME: Full country name.
        ZSCORE_THRESHOLD: Threshold value for anomaly detection using Z-score.
        IQR_MULTIPLIER: Multiplier value for anomaly detection using the
            interquartile range (IQR) method.
        CORRELATION_THRESHOLD: Threshold to identify significant
            correlations between datasets.
        POPULATION_AGE_YOUNG: Age defining the younger population boundary.
        POPULATION_AGE_OLD: Age defining the older population boundary.
        GDP_BASE_YEAR: Base year for GDP calculations.
        DIRECTORY_STRUCTURE: A dictionary specifying paths for input, output,
            processed, enriched, analysis, anomalies, visualizations,
            and logs directories.
    """

    COUNTRY_CODE: str = "BJ"
    COUNTRY_NAME: str = "Bénin"

    ZSCORE_THRESHOLD: float = 3.0
    IQR_MULTIPLIER: float = 1.5
    CORRELATION_THRESHOLD: float = 0.7

    POPULATION_AGE_YOUNG: int = 15
    POPULATION_AGE_OLD: int = 65
    GDP_BASE_YEAR: int = 2015

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
    """
    Sets up the analysis environment with logging, display configurations for
    pandas, and visualization settings for seaborn and matplotlib. Ensures a
    consistent and user-friendly environment for data analysis and visualization.

    Raises:
        OSError: If the specified log directory cannot be created or accessed.

    Parameters:
        log_dir (Optional[Path]): Path to the directory for saving log files.
                                  If None, logging to a file is skipped.
    """
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
    """
    Handles directory structure management and initialization.

    DirectoryManager is a utility class designed to manage and organize
    a set of directories based on a predefined structure. It allows for
    initializing directory structures, retrieving directory paths by
    name, and logging activity. It can be configured using an AnalysisConfig
    object or defaults to a predefined configuration.

    Attributes:
        base_dir (Path): The base directory where the directory structure
            will be created. Defaults to the current working directory if
            not specified.
        config (AnalysisConfig): The configuration object containing the
            directory structure specifications.
        logger (Logger): Logger instance for logging activities.
    """

    def __init__(
        self, base_dir: Optional[Path] = None, config: Optional[AnalysisConfig] = None
    ):
        self.base_dir = base_dir or Path(".")
        self.config = config or AnalysisConfig()
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
@dataclass
class AnomalyReport:
    """
    Represents a report for detected anomalies in a dataset.

    This class encapsulates details about anomalies identified in a specific dataset
    and variable. It provides a structure to store the summary of anomalies including
    anomaly count, percentage, type, and additional details. The timestamp is
    automatically captured when an object of this class is instantiated.

    Attributes:
        dataset_name: Name of the dataset where anomalies are detected.
        variable: The specific variable in the dataset associated with the anomalies.
        anomaly_type: Type of the anomaly identified (e.g., outlier, missing data).
        anomaly_count: Total number of anomalies detected.
        anomaly_percentage: Percentage of anomalies relative to the total data points.
        details: Additional information about the anomalies.
        timestamp: The time and date when the anomaly report was created.

    Methods:
        to_dict:
            Converts the anomaly report instance into a dictionary representation.
            The dictionary includes dataset name, variable, anomaly type, count,
            percentage, details, and timestamp for easier serialization or logging.
    """
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
    """
    Class responsible for loading datasets from a specified directory of CSV files.

    This class provides functionality for reading multiple CSV files in a given directory and
    parsing them into pandas DataFrame objects. The primary purpose of the class is to facilitate
    organized dataset loading along with maintaining logging for errors, warnings, and processing
    information. It supports UTF-8 encoded files and processes only files with a .csv extension,
    skipping any empty files.
    """

    def __init__(self, input_dir: Path):
        self.input_dir = input_dir
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
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
    """Performs comprehensive analysis of datasets, including descriptive analytics,
    temporal and spatial variable identification, and summary reports.

    This class is designed to provide an in-depth look into the structure and
    content of datasets. It helps pinpoint missing data, identify variable types,
    and generate summarized views of analytic results. Primarily used for
    exploratory data analysis, the class includes methods for handling datasets
    with temporal and spatial attributes. The results are returned in structured
    formats for downstream usage.

    Attributes:
        config (Optional[AnalysisConfig]): Configuration object for specifying
            analysis settings.
    """
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    def analyze_dataset(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        self.logger.info(f"📊 Analyse descriptive: {dataset_name}")

        analysis: Dict[str, Any] = {
            "dataset_name": dataset_name,
            "shape": df.shape,
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024**2), 2),
            "columns": list(df.columns),
            "dtypes": df.dtypes.value_counts().to_dict(),
        }

        analysis["completeness"] = {
            "total_cells": df.size,
            "non_null_cells": df.notna().sum().sum(),
            "null_cells": df.isna().sum().sum(),
            "completeness_rate": round(df.notna().sum().sum() / df.size * 100, 2),
        }

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
    """
    TrendAnalyzer

    The TrendAnalyzer class provides tools to analyze temporal trends and calculate
    growth rates from data. It is designed to operate on pandas DataFrame objects
    and assists in processing time-series data for meaningful insights. It features
    capabilities such as temporal aggregation, growth rate computations, and trend
    analysis across multiple value columns.

    Attributes:
        config (Optional[AnalysisConfig]): Configuration object for analysis.
        logger (logging.Logger): Logger instance for the class, utilized for
        logging informational messages and warnings.
    """
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    def analyze_temporal_trends(
        self, df: pd.DataFrame, time_col: str, value_cols: List[str]
    ) -> Dict[str, pd.DataFrame]:
        self.logger.info(f"📈 Analyse des tendances temporelles")

        trends = {}

        for value_col in value_cols:
            if value_col not in df.columns or not pd.api.types.is_numeric_dtype(
                df[value_col]
            ):
                continue

            try:
                trend_df = (
                    df.groupby(time_col)[value_col]
                    .agg(["count", "mean", "median", "std", "min", "max"])
                    .reset_index()
                )

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
        result = df.copy()
        result = result.sort_values(time_col)

        result["growth_rate"] = result[value_col].pct_change() * 100

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
    """
    Handles spatial data analysis tasks related to distributions and regional disparities.

    The SpatialAnalyzer class provides methods for analyzing spatial distributions within a
    DataFrame and calculating various metrics to identify regional disparities. It enables
    users to compute descriptive statistics and evaluate inequality measures such as the Gini
    coefficient for spatially grouped datasets.
    """
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    def analyze_spatial_distribution(
        self, df: pd.DataFrame, spatial_col: str, value_cols: List[str]
    ) -> pd.DataFrame:
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
    """
    Analyzes correlations within datasets and across multiple datasets.

    The CorrelationAnalyzer class provides utilities for calculating correlation matrices,
    identifying strong correlations, and analyzing relationships between datasets. It
    supports various correlation methods and can work with user-defined configurations.

    Attributes:
        config (AnalysisConfig): Configuration used for correlation analysis.
        logger (logging.Logger): Logger for tracking analysis process logs.

    Methods:
        calculate_correlations: Computes the correlation matrix for the numeric columns in a dataset.
        find_strong_correlations: Identifies strong correlations meeting or exceeding a threshold
                                   within a correlation matrix.
        cross_dataset_correlation: Analyzes correlations between two datasets by merging them
                                    on specified columns and examining their numeric columns.
    """
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    def calculate_correlations(
        self, df: pd.DataFrame, method: str = "pearson"
    ) -> pd.DataFrame:
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
    """
    Class for detecting anomalies in data using statistical methods.

    This class provides methods to identify anomalies in datasets by applying Z-score and
    IQR-based anomaly detection techniques. It also detects inconsistencies in data, such
    as negative values in numerical columns or invalid year values outside a specified range.
    The class is designed to handle datasets in the form of pandas DataFrames and can generate
    reports summarizing detected anomalies.

    Attributes:
        config (AnalysisConfig): Configuration settings for the anomaly detection process.
        anomaly_reports (List[AnomalyReport]): List of anomaly reports generated during
            the detection process.
    """
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.anomaly_reports: List[AnomalyReport] = []

    def detect_zscore_anomalies(
        self, df: pd.DataFrame, dataset_name: str
    ) -> pd.DataFrame:
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
        inconsistencies = []

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
        if not self.anomaly_reports:
            return pd.DataFrame()

        return pd.DataFrame([report.to_dict() for report in self.anomaly_reports])


# %%
class IndicatorBuilder:
    """
    This class is responsible for constructing various socio-economic statistical indicators.

    The IndicatorBuilder class provides methods to generate demographic, economic, education,
    and composite indicators from a provided dataset. These indicators can be further used
    for statistical analysis, regional development monitoring, or economic assessments. The
    class allows customization through configuration settings.

    Attributes:
        config (AnalysisConfig): Configuration object for the builder, including settings
            like base year for GDP calculations.
        logger (Logger): Logger instance for capturing events during the indicator-building
            process.
    """
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    def build_demographic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        if "population" in df.columns or "total_population" in df.columns:
            pop_col = "population" if "population" in df.columns else "total_population"

            if "year" in df.columns or "année" in df.columns:
                time_col = "year" if "year" in df.columns else "année"
                result = result.sort_values(time_col)
                result["population_growth_rate"] = result[pop_col].pct_change() * 100

        if "population_0_14" in df.columns and "total_population" in df.columns:
            result["youth_population_ratio"] = (
                result["population_0_14"] / result["total_population"]
            ) * 100

        if "population_65_plus" in df.columns and "total_population" in df.columns:
            result["elderly_population_ratio"] = (
                result["population_65_plus"] / result["total_population"]
            ) * 100

        if all(col in df.columns for col in ["total_population", "surface_area_km2"]):
            result["population_density"] = (
                result["total_population"] / result["surface_area_km2"]
            )

        self.logger.info("✅ Indicateurs démographiques créés")
        return result

    def build_economic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        if all(col in df.columns for col in ["gdp", "total_population"]):
            result["gdp_per_capita"] = result["gdp"] / result["total_population"]

        if "gdp" in df.columns:
            if "year" in df.columns or "année" in df.columns:
                time_col = "year" if "year" in df.columns else "année"
                result = result.sort_values(time_col)
                result["gdp_growth_rate"] = result["gdp"].pct_change() * 100

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
        result = df.copy()

        if all(
            col in df.columns for col in ["enrolled_students", "school_age_population"]
        ):
            result["net_enrollment_rate"] = (
                result["enrolled_students"] / result["school_age_population"]
            ) * 100

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
        result = df.copy()

        available_indicators = [ind for ind in indicators if ind in df.columns]

        if not available_indicators:
            self.logger.warning(f"⚠️ Aucun indicateur disponible pour {index_name}")
            return result

        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(df[available_indicators].fillna(0))
        normalized_df = pd.DataFrame(
            normalized_data, columns=available_indicators, index=df.index
        )

        if weights is None:
            weights = [1 / len(available_indicators)] * len(available_indicators)

        result[index_name] = sum(
            normalized_df[ind] * weight
            for ind, weight in zip(available_indicators, weights)
        )

        result[index_name] = (
            (result[index_name] - result[index_name].min())
            / (result[index_name].max() - result[index_name].min())
        ) * 100

        self.logger.info(f"✅ Indice composite '{index_name}' créé")
        return result

    def build_regional_development_index(self, df: pd.DataFrame) -> pd.DataFrame:
        indicators = []

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
    """
    Facilitates various types of data aggregation, including temporal, spatial, normalization
    by population, and multi-level aggregation.

    The class provides methods to manipulate, aggregate, and normalize data in structured
    tabular formats, such as those handled by pandas DataFrames. Aggregation can be performed
    based on specific time, spatial, or hierarchical grouping levels, with support for custom
    aggregation functions. Additionally, normalization of values by population is available
    to standardize metrics for better interpretability across datasets.

    Attributes:
        config: Optional AnalysisConfig instance to configure the behavior of the engine.
        logger: Logging instance used for debugging and information purposes.

    """
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
    """
    Handles the visualization of various datasets and trends.

    This class provides methods to create and save visualizations for temporal trends, correlation matrices,
    and spatial distributions. The class is designed to facilitate exploratory data analysis and better
    understanding of data patterns. It takes care of rendering plots and saving them to a specified output
    directory.

    Attributes:
        output_dir (Path): Directory where the plots will be saved.
    """
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.logger = logging.getLogger(self.__class__.__name__)

    def plot_temporal_trends(
        self, trends: Dict[str, pd.DataFrame], time_col: str, save: bool = True
    ) -> None:
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
    """
    Handles orchestration of various data analysis phases including descriptive analysis,
    trend analysis, spatial analysis, correlation analysis, and anomaly detection.

    This class is responsible for coordinating the execution and management of different
    analysis modules. It provides functionalities to analyze datasets in stages,
    producing reports, visualizations, and other results as outputs for each phase. It
    integrates functionalities such as data loading, managing file directories, and
    logging progress during analysis.

    Attributes:
        config (Optional[AnalysisConfig]): Configuration settings for the analysis.
        base_dir (Optional[Path]): Base directory for organization of analysis files and outputs.
        logger: Logger instance for tracking execution details.
        dir_manager: Manages directory structure for analysis input and output files.
        directories: Dictionary containing paths to different directories for input, output, etc.
        loader: Loader module for importing datasets.
        descriptive_analyzer: Module for conducting descriptive statistics analysis.
        trend_analyzer: Module for analyzing temporal trends in data.
        spatial_analyzer: Module for analyzing spatial patterns in data.
        correlation_analyzer: Module for detecting correlations and generating correlation matrices.
        anomaly_detector: Module for identifying anomalies and inconsistencies within datasets.
        indicator_builder: Module for building custom indicators or metrics.
        aggregation_engine: Module for aggregating results and data.
        viz_engine: Visualization engine for generating plots and visual insights.
        results: Stores output of executed analysis stages.
    """
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
        self.logger.info("📊 PHASE 1: ANALYSE DESCRIPTIVE")
        print("\n" + "=" * 80)
        print("📊 PHASE 1: ANALYSE DESCRIPTIVE")
        print("=" * 80)

        analyses = {}

        for name, df in datasets.items():
            self.logger.info(f"▶️ Analyse: {name}")
            analysis = self.descriptive_analyzer.analyze_dataset(df, name)
            analyses[name] = analysis

            if "column_statistics" in analysis:
                stats_path = self.directories["analysis"] / f"{name}_column_stats.csv"
                analysis["column_statistics"].to_csv(stats_path, index=False)

        summary_report = self.descriptive_analyzer.generate_summary_report(analyses)
        summary_path = self.directories["analysis"] / "descriptive_summary.csv"
        summary_report.to_csv(summary_path, index=False)

        print("\n✅ Analyse descriptive terminée")
        print(f"   Datasets analysés: {len(analyses)}")
        print(f"   Rapport: {summary_path.name}")

        return analyses

    def run_trend_analysis(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
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

                corr_path = self.directories["analysis"] / f"{name}_correlations.csv"
                corr_matrix.to_csv(corr_path)

                strong_corr_path = (
                    self.directories["analysis"] / f"{name}_strong_correlations.csv"
                )
                if not correlations[name]["strong_correlations"].empty:
                    correlations[name]["strong_correlations"].to_csv(
                        strong_corr_path, index=False
                    )

                self.viz_engine.plot_correlation_heatmap(corr_matrix)

        print(f"\n✅ Analyse des corrélations terminée")
        print(f"   Datasets analysés: {len(correlations)}")

        return correlations

    def run_anomaly_detection(
        self, datasets: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        self.logger.info("🔍 PHASE 5: DÉTECTION DES ANOMALIES")
        print("\n" + "=" * 80)
        print("🔍 PHASE 5: DÉTECTION DES ANOMALIES")
        print("=" * 80)

        all_anomalies = {}
        all_inconsistencies = []

        for name, df in datasets.items():
            self.anomaly_detector.detect_zscore_anomalies(df, name)

            iqr_anomalies = self.anomaly_detector.detect_iqr_anomalies(df, name)

            inconsistencies = self.anomaly_detector.detect_inconsistencies(df, name)

            if not iqr_anomalies.empty:
                all_anomalies[name] = iqr_anomalies

            if inconsistencies:
                all_inconsistencies.extend(inconsistencies)

        anomaly_report = self.anomaly_detector.generate_anomaly_report()
        if not anomaly_report.empty:
            anomaly_path = self.directories["anomalies"] / "anomaly_report.csv"
            anomaly_report.to_csv(anomaly_path, index=False)
            print(f"\n📄 Rapport d'anomalies: {anomaly_path.name}")
            print(f"   Total anomalies détectées: {len(anomaly_report)}")

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
        self.logger.info("🔧 PHASE 6: CRÉATION D'INDICATEURS DÉRIVÉS")
        print("\n" + "=" * 80)
        print("🔧 PHASE 6: CRÉATION D'INDICATEURS DÉRIVÉS")
        print("=" * 80)

        enriched_datasets = {}

        for name, df in datasets.items():
            enriched = df.copy()

            enriched = self.indicator_builder.build_demographic_indicators(enriched)

            enriched = self.indicator_builder.build_economic_indicators(enriched)

            enriched = self.indicator_builder.build_education_indicators(enriched)

            enriched = self.indicator_builder.build_regional_development_index(enriched)

            new_cols = set(enriched.columns) - set(df.columns)

            if new_cols:
                enriched_datasets[name] = enriched

                enriched_path = self.directories["enriched"] / f"{name}_enriched.csv"
                enriched.to_csv(enriched_path, index=False)

                print(f"\n✅ {name}: {len(new_cols)} nouveaux indicateurs")
                for col in sorted(new_cols):
                    print(f"   - {col}")

        print(f"\n✅ Création d'indicateurs terminée")
        print(f"   Datasets enrichis: {len(enriched_datasets)}")

        return enriched_datasets

    def run_aggregations(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        self.logger.info("📊 PHASE 7: AGRÉGATIONS TEMPORELLES ET SPATIALES")
        print("\n" + "=" * 80)
        print("📊 PHASE 7: AGRÉGATIONS TEMPORELLES ET SPATIALES")
        print("=" * 80)

        aggregations = {}

        for name, df in datasets.items():
            dataset_aggs = {}

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
        self.logger.info("📝 Génération de la documentation méthodologique")

        methodology = []

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

        method_path = self.directories["analysis"] / "methodology_documentation.csv"
        methodology_df.to_csv(method_path, index=False)

        self.logger.info(f"✅ Documentation: {method_path.name}")

        return methodology_df

    def run_complete_analysis(self) -> Dict[str, Any]:
        print("\n" + "=" * 80)
        print("🚀 DÉMARRAGE PIPELINE ANALYSE COMPLÈTE - TÂCHE 2")
        print("=" * 80)

        datasets = self.loader.load_all_datasets()

        if not datasets:
            self.logger.error(
                "❌ Aucune donnée chargée. Vérifiez le répertoire d'entrée."
            )
            return {}

        descriptive_results = self.run_descriptive_analysis(datasets)

        trend_results = self.run_trend_analysis(datasets)

        spatial_results = self.run_spatial_analysis(datasets)

        correlation_results = self.run_correlation_analysis(datasets)

        anomaly_results = self.run_anomaly_detection(datasets)

        enriched_datasets = self.run_indicator_creation(datasets)

        aggregation_results = self.run_aggregations(
            enriched_datasets if enriched_datasets else datasets
        )

        methodology_doc = self.generate_methodology_document()

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
    """
    Main function to set up the environment, initialize the orchestrator, and execute
    a complete analysis.

    Return:
        The results of the complete analysis.

    """
    setup_analysis_environment(log_dir=Path("logs_task_2"))

    orchestrator = ExplorationOrchestrator()

    results = orchestrator.run_complete_analysis()

    return results


# %%
_ = main()
