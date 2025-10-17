#%%
import functools
import sys
import time
import warnings
import logging
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any, Union, Callable, Iterable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from urllib.parse import urlparse
from logging.handlers import RotatingFileHandler

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from tqdm import tqdm

#%%
@dataclass
class GlobalConfig:
    """
    Represents the global configuration for a data collection, processing, and API interactions system.

    This class defines several constants and configurations necessary for interacting with various APIs,
    structuring directories, and specifying default indicators for data extraction. It is useful for maintaining
    a unified and centralized configuration setup, enabling seamless adjustments and reducing code redundancy.

    Attributes:
        COUNTRY_CODE (str): The ISO country code.
        COUNTRY_NAME (str): Full name of the country.
        START_YEAR (int): Start year for data collection.
        END_YEAR (int): End year for data collection.
        WORLD_BANK_API_URL (str): Base URL for the World Bank API.
        INSTAD_API_URL (str): Base URL for Instad APIs.
        OVERPASS_API_URL (str): Base URL for Overpass API.
        FMI_API_URL (str): Base URL for the IMF API.
        OMS_API_URL (str): Base URL for the WHO API.
        UNDP_API_URL (str): URL for UNDP indicators data.
        DEFAULT_PER_PAGE (int): Default number of results per page for API and data requests.
        REQUEST_TIMEOUT (int): Timeout duration in seconds for API requests.
        RETRY_COUNT (int): Number of retry attempts for failed API requests.
        DELAY_BETWEEN_REQUESTS (float): Delay in seconds between consecutive API calls.
        DIRECTORY_STRUCTURE (Dict[str, str]): Directory structure configuration dict with specified keys like
            'data', 'raw', 'processed', etc.
        DEFAULT_WB_INDICATORS (List[str]): Default World Bank economic and population indicators.
        DEFAULT_IMF_INDICATORS (List[str]): Default IMF economic indicators.
        DEFAULT_HEALTH_INDICATORS (List[str]): Default WHO health-related indicators.
        OSM_ADMIN_LEVELS (Dict[str, str]): Administrative levels mapping for OSM data.
        EXTERNAL_SCRAPING_URLS (Dict[str, str]): URLs for external scraping resources.
        EXTERNAL_CSV_URLS (List[str]): External CSV file data source URLs.

    Methods:
        __post_init__(): Validates the initial configuration parameters to ensure consistency and correctness.

    Raises:
        ValueError: Raised if START_YEAR is greater than END_YEAR, if the years are outside the acceptable range
            of 1900 to 2100, if RETRY_COUNT is less than 1, or REQUEST_TIMEOUT is less than 1.
    """
    COUNTRY_CODE: str = "BJ"
    COUNTRY_NAME: str = "Bénin"
    START_YEAR: int = 2015
    END_YEAR: int = 2024

    WORLD_BANK_API_URL: str = "https://api.worldbank.org/v2"
    INSTAD_API_URL: str = "https://instad.bj"
    OVERPASS_API_URL: str = "https://overpass-api.de/api/interpreter"
    FMI_API_URL: str = "https://www.imf.org/external/datamapper/api/v1"
    OMS_API_URL: str = "https://ghoapi.azureedge.net/api"
    UNDP_API_URL: str = (
        "https://hdr.undp.org/sites/default/files/2021-22_HDR/HDR21-22_Composite_indices_complete_time_series.csv"
    )

    DEFAULT_PER_PAGE: int = 100
    REQUEST_TIMEOUT: int = 30
    RETRY_COUNT: int = 3
    DELAY_BETWEEN_REQUESTS: float = 0.5

    DIRECTORY_STRUCTURE: Dict[str, str] = field(
        default_factory=lambda: {
            "data": "data_task_1",
            "raw": "data_task_1/raw",
            "processed": "data_task_1/processed",
            "final_data": "data_task_1/final_data",
            "logs": "logs",
            "exports": "exports",
            "docs": "docs",
        }
    )

    DEFAULT_WB_INDICATORS: List[str] = field(
        default_factory=lambda: [
            "SP.POP.TOTL",
            "NY.GDP.MKTP.CD",
            "NY.GDP.PCAP.CD",
            "SE.PRM.NENR",
            "SH.DYN.MORT",
            "AG.LND.TOTL.K2",
            "SL.TLF.TOTL.IN",
            "SP.DYN.TFRT.IN",
        ]
    )

    DEFAULT_IMF_INDICATORS: List[str] = field(
        default_factory=lambda: [
            "NGDP_R",
            "NGDPD",
            "PCPIPCH",
            "LUR",
            "GGX_NGDP",
        ]
    )

    DEFAULT_HEALTH_INDICATORS: List[str] = field(
        default_factory=lambda: [
            "WHOSIS_000001",
            "MDG_0000000001",
            "MDG_0000000003",
            "WHS4_544"
        ]
    )

    OSM_ADMIN_LEVELS: Dict[str, str] = field(
        default_factory=lambda: {"pays": "2", "département": "4", "commune": "6"}
    )

    EXTERNAL_SCRAPING_URLS: Dict[str, str] = field(
        default_factory=lambda: {
            "rgph": "https://www.insae-bj.org/recensement-population.html",
            "edc": "https://www.insae-bj.org/statistiques-economiques.html",
            "emicov": "https://www.insae-bj.org/emicov.html",
        }
    )

    EXTERNAL_CSV_URLS: List[str] = field(
        default_factory=lambda: [
            "https://data.uis.unesco.org/medias/education/SDG4.csv",
        ]
    )

    def __post_init__(self):
        if self.START_YEAR > self.END_YEAR:
            raise ValueError(
                f"START_YEAR ({self.START_YEAR}) must be <= END_YEAR ({self.END_YEAR})"
            )
        if not (1900 <= self.START_YEAR <= 2100 and 1900 <= self.END_YEAR <= 2100):
            raise ValueError("Years must be between 1900 and 2100")
        if self.RETRY_COUNT < 1:
            raise ValueError("RETRY_COUNT must be >= 1")
        if self.REQUEST_TIMEOUT < 1:
            raise ValueError("REQUEST_TIMEOUT must be >= 1")
#%%
@dataclass
class CleaningReport:
    """
    Represents a report detailing data cleaning and processing performed on a dataset.

    This class is designed to store information about various cleaning operations
    performed on dataset columns and rows, such as handling nulls, removing duplicates,
    standardizing columns, converting data types, among others. It provides an organized
    way to track the source of the dataset, the state of rows before and after cleaning,
    and other specific actions taken during the cleaning process. Additionally, the class
    offers a method for converting the cleaning report into a dictionary format.

    Attributes:
        source (str): The origin or source of the dataset.
        initial_rows (int): The number of rows in the dataset before cleaning.
        final_rows (int): The number of rows in the dataset after cleaning.
        rows_removed (int): Total rows removed during the cleaning process. Defaults to 0.
        duplicates_removed (int): Total duplicate rows removed. Defaults to 0.
        nulls_handled (int): Total number of null values handled. Defaults to 0.
        outliers_removed (int): Total outliers removed from the dataset. Defaults to 0.
        columns_standardized (List[str]): List of columns that were standardized. Defaults to an empty list.
        columns_dropped (List[str]): List of columns dropped during cleaning. Defaults to an empty list.
        data_types_converted (Dict[str, str]): Dictionary of columns whose data types were converted,
            mapping column names to their new data types. Defaults to an empty dictionary.
        issues_detected (List[str]): List of issues or problems detected during cleaning. Defaults to an empty list.
        cleaning_timestamp (datetime): The timestamp when cleaning was conducted. Defaults to the current time.

    Methods:
        to_dict:
            Converts the cleaning report into a dictionary representation for easier
            usage in logging, reporting, or further data analysis.

            Returns:
                Dict[str, Any]: A dictionary containing key information of the cleaning
                report, including derived metrics like removal percentage and counts
                of operations performed.
    """
    source: str
    initial_rows: int
    final_rows: int
    rows_removed: int = 0
    duplicates_removed: int = 0
    nulls_handled: int = 0
    outliers_removed: int = 0
    columns_standardized: List[str] = field(default_factory=list)
    columns_dropped: List[str] = field(default_factory=list)
    data_types_converted: Dict[str, str] = field(default_factory=dict)
    issues_detected: List[str] = field(default_factory=list)
    cleaning_timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "initial_rows": self.initial_rows,
            "final_rows": self.final_rows,
            "rows_removed": self.rows_removed,
            "removal_percentage": (
                round((self.rows_removed / self.initial_rows * 100), 2)
                if self.initial_rows > 0
                else 0
            ),
            "duplicates_removed": self.duplicates_removed,
            "nulls_handled": self.nulls_handled,
            "outliers_removed": self.outliers_removed,
            "columns_standardized": len(self.columns_standardized),
            "columns_dropped": len(self.columns_dropped),
            "types_converted": len(self.data_types_converted),
            "issues_count": len(self.issues_detected),
            "timestamp": self.cleaning_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        }
#%%
@dataclass
class PerformanceMetrics:
    """
    Encapsulates performance metrics for tracking the execution of an operation.

    This class provides attributes to measure and record the performance of an operation, such
    as start and end times, duration, the number of items processed, and whether the operation
    succeeded or failed. It includes utility methods for finalizing and converting recorded
    metrics into a dictionary for further use or reporting.

    Attributes:
        operation_name (str): The name of the operation being tracked.
        start_time (datetime): The starting time of the operation.
        end_time (Optional[datetime]): The ending time of the operation. Defaults to None.
        duration_seconds (float): The total duration of the operation in seconds. Defaults
            to 0.0.
        items_processed (int): The number of items processed during the operation. Defaults to 0.
        success (bool): Indicates whether the operation was successful. Defaults to True.
        error_message (Optional[str]): An optional message describing any error that occurred
            during the operation. Defaults to None.
        metadata (Dict[str, Any]): Additional information related to the operation, stored
            as key-value pairs. Defaults to an empty dictionary.

    Methods:
        finalize(items: int = 0, success: bool = True, error: Optional[str] = None): Finalizes
            the metrics by recording the end time, calculating the duration, and updating the
            number of items, success flag, and error message.
        to_dict() -> Dict[str, Any]: Converts the metrics to a dictionary format for reporting
            or storage purposes.
        __str__() -> str: Returns a string representation of the metrics for easy readability.
    """
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    items_processed: int = 0
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def finalize(
            self, items: int = 0, success: bool = True, error: Optional[str] = None
    ):
        self.end_time = datetime.now()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.items_processed = items
        self.success = success
        self.error_message = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation_name,
            "start": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end": (
                self.end_time.strftime("%Y-%m-%d %H:%M:%S") if self.end_time else None
            ),
            "duration": round(self.duration_seconds, 3),
            "items": self.items_processed,
            "throughput": (
                round(self.items_processed / self.duration_seconds, 2)
                if self.duration_seconds > 0
                else 0
            ),
            "success": self.success,
            "error": self.error_message,
            **self.metadata,
        }

    def __str__(self) -> str:
        status = "✅" if self.success else "❌"
        duration_str = f"{self.duration_seconds:.3f}s"
        if self.items_processed > 0:
            throughput = (
                self.items_processed / self.duration_seconds
                if self.duration_seconds > 0
                else 0
            )
            return f"{status} {self.operation_name} | Durée: {duration_str} | Items: {self.items_processed} | Débit: {throughput:.2f} items/s"
        return f"{status} {self.operation_name} | Durée: {duration_str}"
#%%
def is_script() -> bool:
    """
    Determines if the current executing script is being run as a standalone script.

    This function checks if the module `__main__` (the entry point of the program) has
    an attribute `__file__`, which indicates whether it is executed as a script or an
    interactive session. This can be useful to distinguish between these two modes of
    execution in Python.

    Returns:
        bool: True if the script is being executed as a standalone script, False otherwise.
    """
    return hasattr(sys.modules["__main__"], "__file__")
#%%
def setup_environment(
        log_dir: Optional[Path] = None, log_level: int = logging.INFO
) -> None:
    """
    Sets up the environment by configuring logging, warnings, pandas options, and visualization settings.

    This function handles the following:
    - Configures the logging system to output messages to the console and optionally to a file.
    - Suppresses deprecated, user, and future warnings.
    - Adjusts pandas settings for data display, including maximum rows, column limits, and formatting.
    - Customizes Seaborn and Matplotlib aesthetics for visualizations.

    It is useful for initializing a consistent environment for data analysis or application workflows.

    Parameters:
    log_dir: Optional[Path]
        Path to the directory where log files should be stored. If not provided, logging to
        a file is disabled.
    log_level: int
        The logging level used to filter messages. Defaults to logging.INFO.

    Returns:
    None

    Raises:
    None
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="bs4")
    warnings.filterwarnings("ignore", category=FutureWarning)

    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format, date_format))

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[console_handler],
    )

    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "system.log", maxBytes=2_000_000, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        logging.getLogger().addHandler(file_handler)

    pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.precision", 2)

    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (12, 8),
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )
    sns.set_palette("Set2")
#%%
class DirectoryManager:
    """
    Manages directories and provides utility methods for structure initialization
    and path retrieval.

    The DirectoryManager class is designed to handle the creation and management
    of directory structures. It enables initializing a directory structure based
    on a given configuration, retrieving paths for specific directories, and listing
    all managed directories. This is particularly useful in applications that require
    consistency in directory organization.

    Attributes:
        base_dir (Path): The base directory where the structure will be created.
            Defaults to the current script's directory or the current working
            directory if not specified.
        custom_structure (Optional[Dict[str, str]]): An optional custom directory
            structure defined as a mapping of directory names to relative paths.
    """
    def __init__(
            self,
            base_dir: Optional[Path] = None,
            custom_structure: Optional[Dict[str, str]] = None,
    ):
        self.base_dir = base_dir or (
            Path(__file__).parent if is_script() else Path(".")
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self._directories: Dict[str, Path] = {}
        self.custom_structure = custom_structure

    def initialize_structure_directory(
            self, structure: Optional[Dict[str, str]] = None
    ) -> Dict[str, Path]:
        structure_to_use = (
                structure or self.custom_structure or GlobalConfig().DIRECTORY_STRUCTURE
        )
        for name, path in structure_to_use.items():
            full_path = self.base_dir / path
            full_path.mkdir(parents=True, exist_ok=True)
            self._directories[name] = full_path
        self.logger.info(f"{len(self._directories)} dossiers créés avec succès.")
        return self._directories

    def get_path(self, name: str) -> Optional[Path]:
        path = self._directories.get(name)
        if path is None:
            self.logger.warning(f"Le dossier '{name}' n'existe pas dans la structure.")
        return path

    def list_directories(self) -> Dict[str, Path]:
        return self._directories.copy()
#%%
class AbstractCollector(ABC):
    """
    Abstract base class for data collection and saving operations.

    This class provides a general framework for collecting data from various sources
    and saving it in different formats. It includes methods to validate URLs,
    handle HTTP requests with retries, and save data using multiple formats. The
    class is designed to be extended by specific data collector implementations.

    Attributes:
        config (GlobalConfig): Configuration for the collector, including retry
            counts, timeouts, and delays between requests.
        logger (logging.Logger): Logger instance for recording events and errors.

    Methods:
        collect_data:
            Abstract method that must be implemented by subclasses. Used to perform
            the data collection task.

        save_data:
            Saves a DataFrame in a specified format to the provided file path.

    """
    def __init__(self, config: GlobalConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session = self._create_session()

    @staticmethod
    def _create_session() -> requests.Session:
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Educational Research Bot/1.0)",
                "Accept": "application/json, text/html, */*",
                "Accept-Language": "fr,en;q=0.9",
            }
        )
        return session

    def _validate_url(self, url: str) -> bool:
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception as e:
            self.logger.debug(f"Invalid URL: {e}")
            return False

    def _make_request_with_retry(
            self, url: str, **kwargs
    ) -> Tuple[Optional[requests.Response], bool]:
        if not self._validate_url(url):
            return None, False

        method = kwargs.pop("method", "GET")

        for attempt in range(self.config.RETRY_COUNT):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    timeout=self.config.REQUEST_TIMEOUT,
                    **kwargs,
                )
                response.raise_for_status()
                if self.config.DELAY_BETWEEN_REQUESTS > 0:
                    time.sleep(self.config.DELAY_BETWEEN_REQUESTS)
                return response, True
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response else "N/A"
                self.logger.warning(
                    f"🔄 Attempt {attempt + 1}/{self.config.RETRY_COUNT} - HTTP {status_code} on {url}"
                )
            except requests.exceptions.Timeout:
                self.logger.warning(
                    f"🔄 Attempt {attempt + 1}/{self.config.RETRY_COUNT} - Timeout on {url}"
                )
            except requests.exceptions.ConnectionError:
                self.logger.warning(
                    f"🔄 Attempt {attempt + 1}/{self.config.RETRY_COUNT} - Connection error on {url}"
                )
            except requests.exceptions.RequestException as e:
                self.logger.warning(
                    f"🔄 Attempt {attempt + 1}/{self.config.RETRY_COUNT} - Error on {url}: {e}"
                )

            if attempt < self.config.RETRY_COUNT - 1:
                sleep_time = 2 ** attempt
                self.logger.debug(f"Waiting {sleep_time}s before retry...")
                time.sleep(sleep_time)

        self.logger.error(f"❌ Failed after {self.config.RETRY_COUNT} attempts: {url}")
        return None, False

    @abstractmethod
    def collect_data(self) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        pass

    def save_data(
            self,
            data: pd.DataFrame,
            file_path: Path,
            format_type: str = "csv",
            add_metadata: bool = True,
    ) -> Tuple[bool, Optional[Path]]:
        if not isinstance(file_path, Path):
            raise TypeError("file_path must be a pathlib.Path object")
        if data.empty:
            self.logger.warning("Empty DataFrame, no data to save.")
            return False, None

        try:
            meta_data = data.copy()
            if add_metadata:
                meta_data["collection_timestamp"] = datetime.now()
                meta_data["collector"] = self.__class__.__name__

            format_type = format_type.lower()
            if format_type == "csv":
                meta_data.to_csv(file_path, index=False, encoding="utf-8")
            elif format_type == "excel":
                meta_data.to_excel(file_path, index=False, engine="openpyxl")
            elif format_type == "json":
                meta_data.to_json(
                    file_path,
                    orient="records",
                    force_ascii=False,
                    indent=2,
                    date_format="iso",
                )
            elif format_type == "parquet":
                meta_data.to_parquet(file_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format_type}")

            size_mb = file_path.stat().st_size / (1024 * 1024)
            self.logger.info(
                f"✅ Saved: {file_path.name} ({len(data)} rows, {size_mb:.2f} MB)"
            )
            return True, file_path
        except Exception as e:
            self.logger.error(f"❌ Save error {file_path}: {e}")
            return False, None
#%%
class PerformanceTracker:
    """
    Tracks and summarizes performance metrics for various operations.

    This class is designed to collect and manage performance metrics, which
    can be used to analyze the efficiency and effectiveness of various operations.
    It provides functionalities to add new metrics, summarize them, and print
    a consolidated report. Useful for logging and debugging purposes in scenarios
    where monitoring performance is critical.

    Attributes:
        metrics: List of performance metrics collected, holding objects of type
            PerformanceMetrics.
        logger: Logger instance specifically for the class, used to log internal
            events and debug information.

    Methods:
        add_metric(metric: PerformanceMetrics):
            Adds a new performance metric to the tracker.

        get_summary() -> Dict[str, Any]:
            Returns a summary of collected performance data.

        print_summary():
            Prints the summarized performance data in a formatted view.
    """
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.logger = logging.getLogger(self.__class__.__name__)

    def add_metric(self, metric: PerformanceMetrics) -> None:
        self.metrics.append(metric)
        self.logger.debug(f"{metric.operation_name}: {metric}")

    def get_summary(self) -> Dict[str, Any]:
        if not self.metrics:
            return {"total_operations": 0}
        total_duration = sum(m.duration_seconds for m in self.metrics)
        successful = sum(1 for m in self.metrics if m.success)
        return {
            "total_operations": len(self.metrics),
            "successful": successful,
            "failed": len(self.metrics) - successful,
            "total_duration": round(total_duration, 3),
            "avg_duration": round(total_duration / len(self.metrics), 3),
            "total_items": sum(m.items_processed for m in self.metrics),
        }

    def print_summary(self):
        summary = self.get_summary()
        print("\n" + "=" * 70)
        print("RÉSUMÉ DES PERFORMANCES")
        print("=" * 70)
        for key, value in summary.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        print("=" * 70 + "\n")


_global_tracker = PerformanceTracker()
#%%
def timer(
        operation_name: Optional[str] = None,
        log_result: bool = True,
        track_metrics: bool = True,
):
    """
    Decorator to measure and log the execution time of a function.

    This decorator allows you to track the performance of a function by measuring its
    execution time and, optionally, logging the result and tracking performance metrics.
    It uses the PerformanceMetrics helper to record the operation name, duration, success
    status, and any error that occurred.

    Parameters:
    operation_name: Optional[str]
        The name of the operation to be tracked. If not provided, the module and function
        name of the decorated function will be used.
    log_result: bool
        Determines whether the result of the execution should be logged. Defaults to True.
    track_metrics: bool
        Determines whether the performance metrics should be tracked using a global tracker.
        Defaults to True.

    Returns:
    Callable
        A decorated function that measures and logs its execution time.

    Raises:
    Exception
        Propagates any exception raised by the decorated function.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            op_name = (
                    operation_name
                    or f"{getattr(func, '__module__', '<unknown>')}.{func.__name__}"
            )
            metric = PerformanceMetrics(
                operation_name=op_name, start_time=datetime.now()
            )
            logger = logging.getLogger(getattr(func, "__module__", "<unknown>"))

            if log_result:
                logger.info(f"⏱️  Start: {op_name}")

            try:
                result = func(*args, **kwargs)
                metric.finalize(success=True)
                if log_result:
                    logger.info(f"✅ Done: {op_name} in {metric.duration_seconds:.3f}s")
                if track_metrics:
                    _global_tracker.add_metric(metric)
                return result
            except Exception as e:
                metric.finalize(success=False, error=str(e))
                if log_result:
                    logger.error(
                        f"❌ Failed: {op_name} after {metric.duration_seconds:.3f}s - {e}"
                    )
                if track_metrics:
                    _global_tracker.add_metric(metric)
                raise

        return wrapper

    return decorator
#%%
@contextmanager
def track_progress(
        iterable: Iterable,
        desc: str = "Processing",
        total: Optional[int] = None,
        unit: str = "item",
        leave: bool = True,
        **tqdm_kwargs,
):
    """
    Context manager for tracking progress of iterations using tqdm.

    This context manager wraps an iterable with tqdm to display a progress bar,
    providing real-time feedback on the progress of iteration steps. The optional
    parameters allow customization of the progress bar's display.

    Parameters:
        iterable (Iterable): The iterable whose progress is being tracked.
        desc (str): Description text displayed on the progress bar. Defaults
                    to "Processing".
        total (Optional[int]): The total number of items to iterate over.
                               If None, the iterable length is used if
                               available. Defaults to None.
        unit (str): Unit description for each iteration step. Defaults to "item".
        leave (bool): Whether to keep the progress bar after completion.
                      Defaults to True.
        **tqdm_kwargs: Additional keyword arguments to customize tqdm behavior.

    Yields:
        tqdm.std.tqdm: A tqdm progress bar object wrapping the given iterable.
    """
    pbar = tqdm(
        iterable=iterable,
        desc=desc,
        total=total,
        unit=unit,
        leave=leave,
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        **tqdm_kwargs,
    )
    try:
        yield pbar
    finally:
        pbar.close()
#%%
class WorldBankCollector(AbstractCollector):
    """
    Handles World Bank data collection.

    This class is responsible for collecting and processing data from the World Bank API.
    It fetches indicator data for specified countries and years, processes the raw data,
    and aggregates the results into a structured DataFrame. The class structures its
    workflow for retry mechanisms, error logging, and performance metrics tracking.
    """
    @timer(operation_name="WorldBank.fetch_indicator", track_metrics=True)
    def _fetch_indicator_data(
            self,
            indicator: str,
            start_year: Optional[int] = None,
            end_year: Optional[int] = None,
    ) -> pd.DataFrame:
        url = f"{self.config.WORLD_BANK_API_URL}/country/{self.config.COUNTRY_CODE}/indicator/{indicator}"
        params = {
            "date": f"{start_year or self.config.START_YEAR}:{end_year or self.config.END_YEAR}",
            "format": "json",
            "per_page": self.config.DEFAULT_PER_PAGE,
        }
        response, success = self._make_request_with_retry(url, params=params)
        if not success or response is None:
            return pd.DataFrame()

        try:
            data = response.json()
            entries = data[1] if isinstance(data, list) and len(data) > 1 else []
            records = [
                {
                    "indicator_code": entry["indicator"]["id"],
                    "indicator_name": entry["indicator"]["value"],
                    "country_code": entry["country"]["id"],
                    "country_name": entry["country"]["value"],
                    "year": pd.to_numeric(entry["date"], errors="coerce"),
                    "value": pd.to_numeric(entry["value"], errors="coerce"),
                    "source": "World Bank API",
                }
                for entry in entries
            ]
            return pd.DataFrame(records)
        except (ValueError, KeyError, IndexError) as e:
            self.logger.error(f"❌ Parse error {indicator}: {e}")
            return pd.DataFrame()

    @timer(operation_name="WorldBank.collect_all", track_metrics=True)
    def collect_data(self, indicators: Optional[List[str]] = None) -> pd.DataFrame:
        indicators = indicators or self.config.DEFAULT_WB_INDICATORS
        self.logger.info(
            f"🌍 Start World Bank collection ({len(indicators)} indicators)"
        )
        all_data = []
        with track_progress(indicators, desc="World Bank", unit="indicator") as pbar:
            for indicator in pbar:
                pbar.set_postfix_str(f"Indicator: {indicator}")
                df = self._fetch_indicator_data(indicator)
                if not df.empty:
                    all_data.append(df)
                    self.logger.info(f"✅ {len(df)} records collected")
                time.sleep(self.config.DELAY_BETWEEN_REQUESTS)

        result = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
        metric = PerformanceMetrics(
            operation_name="WorldBank.collect_summary",
            start_time=datetime.now(),
            items_processed=len(result),
        )
        metric.finalize(items=len(result))
        _global_tracker.add_metric(metric)
        self.logger.info(f"✅ World Bank done: {len(result)} records")
        return result
#%%
class WebScrapingCollector(AbstractCollector):
    """
    WebScrapingCollector handles the process of scraping tabular data from web pages and consolidating
    it into a structured format.

    This class is designed to perform web scraping for extracting HTML tables from specific URLs and transform them
    into a Pandas DataFrame for further processing. It provides functionality to collect data from multiple sources
    simultaneously and ensures data integrity by cleaning up the tables before they are consolidated into the final
    output. The scraping tasks are tracked with metrics, and progress is logged effectively.

    Attributes:
        logger: Logging utility used for tracking the scraping process.
    """
    @timer(operation_name="WebScraping.scrape_tables", track_metrics=True)
    def _scrape_html_tables(
        self, url: str, source_name: str, max_tables: int = 10
    ) -> pd.DataFrame:
        response, success = self._make_request_with_retry(url)
        if not success or response is None:
            return pd.DataFrame()

        try:
            soup = BeautifulSoup(response.content, "html.parser")
            tables = soup.find_all("table")
            self.logger.info(f"📋 {len(tables)} tables found on {url}")
            
            scraped_data = []
            for i, table in enumerate(tables[:max_tables]):
                try:
                    df = pd.read_html(str(table), header=0)[0]
                    
                    if df.empty or len(df.columns) < 2:
                        continue
                    
                    df = df.dropna(how='all', axis=0)
                    df = df.dropna(how='all', axis=1)
                    
                    if df.empty:
                        continue
                    
                    df.columns = [str(col).strip() for col in df.columns]
                    
                    df['source_url'] = url
                    df['source_name'] = source_name
                    df['table_index'] = i
                    df['extraction_date'] = datetime.now()
                    
                    scraped_data.append(df)
                    
                except Exception as e:
                    self.logger.debug(f"Skipping table {i}: {e}")
                    continue
            
            if not scraped_data:
                self.logger.warning(f"No valid tables extracted from {url}")
                return pd.DataFrame()
            
            return pd.concat(scraped_data, ignore_index=True)
            
        except Exception as e:
            self.logger.error(f"❌ Scraping error {url}: {e}")
            return pd.DataFrame()

    @timer(operation_name="WebScraping.collect_all", track_metrics=True)
    def collect_data(self, max_tables: int = 10) -> pd.DataFrame:
        self.logger.info("🕷️ Start web scraping")
        urls = {
            "instad_trimestres": "https://instad.bj/publications/publications-trimestrielles",
            "instad_mensuelles": "https://instad.bj/publications/publications-mensuelles",
        }
        all_data = []
        with track_progress(
            urls.items(), desc="Web Scraping", total=len(urls), unit="site"
        ) as pbar:
            for source_name, url in pbar:
                pbar.set_postfix_str(f"Source: {source_name}")
                df = self._scrape_html_tables(url, source_name, max_tables)
                if not df.empty:
                    all_data.append(df)
        
        result = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
        self.logger.info(f"✅ Scraping done: {len(result)} records")
        return result
#%%
class GeographicCollector(AbstractCollector):
    """
    Collects and processes geographic data using Overpass API.

    This class is responsible for collecting geographic information such as cities
    and administrative boundaries. It connects to the Overpass API, runs specific
    queries, processes the responses, and returns the results in a structured
    format. The primary goal is to gather location-based data relevant to a given
    region or country for further analysis.

    Methods:
        collect_data: Initiates the collection process for cities and administrative
        boundaries and organizes the data into structured outputs.

    """
    @timer(operation_name="Geographic.execute_query", track_metrics=True)
    def _execute_overpass_query(self, query: str, data_type: str) -> pd.DataFrame:
        response, success = self._make_request_with_retry(
            self.config.OVERPASS_API_URL, data={"data": query}, method="POST"
        )
        if not success or response is None:
            return pd.DataFrame()

        try:
            data = response.json()
            elements = data.get("elements", [])
            records = []
            for element in elements:
                if "tags" not in element:
                    continue
                record = {
                    "name": element["tags"].get("name"),
                    "osm_id": element.get("id"),
                    "latitude": element.get("lat")
                                or element.get("center", {}).get("lat"),
                    "longitude": element.get("lon")
                                 or element.get("center", {}).get("lon"),
                    "data_type": data_type,
                    "source": "OpenStreetMap",
                }
                if data_type == "cities":
                    record.update(
                        {
                            "place_type": element["tags"].get("place"),
                            "population": pd.to_numeric(
                                element["tags"].get("population"), errors="coerce"
                            ),
                        }
                    )
                else:
                    record.update(
                        {
                            "admin_level": element["tags"].get("admin_level"),
                            "wikidata": element["tags"].get("wikidata"),
                        }
                    )
                records.append(record)
            self.logger.info(f"📍 {len(records)} {data_type} elements collected")
            return pd.DataFrame(records)
        except (ValueError, KeyError) as e:
            self.logger.error(f"❌ Parse error Overpass {data_type}: {e}")
            return pd.DataFrame()

    @timer(operation_name="Geographic.collect_all", track_metrics=True)
    def collect_data(self) -> Dict[str, pd.DataFrame]:
        self.logger.info("🗺️ Start geographic collection")
        results = {}
        cities_query = f"""
        [out:json][timeout:60];
        area["ISO3166-1"="{self.config.COUNTRY_CODE}"];
        (node(area)["place"~"city|town|village"]; way(area)["place"~"city|town|village"];);
        out center tags;
        """
        cities_df = self._execute_overpass_query(cities_query, "cities")
        if not cities_df.empty:
            results["cities"] = cities_df

        admin_levels = list(self.config.OSM_ADMIN_LEVELS.items())
        with track_progress(
                admin_levels, desc="Admin boundaries", unit="level"
        ) as pbar:
            for level_name, level_code in pbar:
                pbar.set_postfix_str(f"Level: {level_name}")
                admin_query = f"""
                [out:json][timeout:60];
                relation["boundary"="administrative"]["admin_level"="{level_code}"]["name"~"{self.config.COUNTRY_NAME}|Benin"];
                out center tags;
                """
                admin_df = self._execute_overpass_query(
                    admin_query, f"admin_{level_name}"
                )
                if not admin_df.empty:
                    results[f"admin_{level_name}"] = admin_df

        total = sum(len(df) for df in results.values())
        self.logger.info(f"✅ Geographic done: {total} records")
        return results
#%%
class ExternalCollector(AbstractCollector):
    """
    Defines the ExternalCollector class, which is responsible for collecting and processing external data
    sources. This class fetches content from specified URLs, processes it into structured data formats,
    and normalizes it for downstream analysis. It provides functionality to handle multiple URLs and
    combines the data into a unified result.

    Attributes:
        config: Configuration object containing external data source URLs.
        logger: Logger instance to record activities and errors during data collection.

    Parameters:
        AbstractCollector: Base class, which ExternalCollector extends functionality from.
    """
    @timer(operation_name="External.download_file", track_metrics=True)
    def _download_data(self, url: str) -> pd.DataFrame:
        response, success = self._make_request_with_retry(url)
        if not success or response is None:
            return pd.DataFrame()

        try:
            from io import BytesIO

            content = response.content
            try:
                df = pd.read_csv(BytesIO(content))
                df["source"] = url
                return df
            except pd.errors.EmptyDataError:
                self.logger.warning(f"⚠️ Empty file: {url}")
                return pd.DataFrame()
            except Exception:
                pass

            try:
                import json

                json_data = json.loads(content)
                df = pd.json_normalize(json_data)
                df["source"] = url
                return df
            except Exception as e:
                self.logger.error(f"❌ Unsupported format for {url}: {e}")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"❌ Download error {url}: {e}")
            return pd.DataFrame()

    @timer(operation_name="External.collect_all", track_metrics=True)
    def collect_data(self, urls: Optional[List[str]] = None) -> pd.DataFrame:
        urls = urls or self.config.EXTERNAL_CSV_URLS
        self.logger.info(f"🌐 External collection ({len(urls)} sources)")
        all_data = []
        with track_progress(
                enumerate(urls, 1), desc="External sources", total=len(urls), unit="source"
        ) as pbar:
            for i, url in pbar:
                pbar.set_postfix_str(f"URL {i}/{len(urls)}")
                df = self._download_data(url)
                if not df.empty:
                    all_data.append(df)
                    self.logger.info(f"✅ {len(df)} records")
        result = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
        self.logger.info(f"✅ External done: {len(result)} records")
        return result
#%%
class IMFCollector(AbstractCollector):
    """
    Collector class for fetching and processing IMF indicator data.

    This class is responsible for collecting data from the IMF API based on
    specified indicators, handling API responses and errors, and transforming
    the data into a pandas DataFrame for further use. It supports retrying
    failed requests, tracking performance metrics, and logging the progress of
    the data collection process.

    Methods
    -------
    _fetch_indicator_data(indicator: str) -> pd.DataFrame
        Retrieves a single indicator's data from the IMF API and processes
        it into a pandas DataFrame.

    collect_data(indicators: Optional[List[str]] = None) -> pd.DataFrame
        Collects data for multiple indicators from the IMF API and aggregates
        them into a pandas DataFrame.
    """
    @timer(operation_name="IMF.fetch_indicator", track_metrics=True)
    def _fetch_indicator_data(self, indicator: str) -> pd.DataFrame:
        url = f"{self.config.FMI_API_URL}/{indicator}"
        response, success = self._make_request_with_retry(url)
        if not success or response is None:
            return pd.DataFrame()

        try:
            data = response.json()
            records = []
            if isinstance(data, dict) and "values" in data:
                country_data = (
                    data["values"].get(indicator, {}).get(self.config.COUNTRY_CODE, {})
                )
                for year, value in country_data.items():
                    records.append(
                        {
                            "indicator_code": indicator,
                            "country_code": self.config.COUNTRY_CODE,
                            "year": pd.to_numeric(year, errors="coerce"),
                            "value": pd.to_numeric(value, errors="coerce"),
                            "source": "IMF",
                        }
                    )
            return pd.DataFrame(records)
        except Exception as e:
            self.logger.error(f"❌ Parse error IMF {indicator}: {e}")
            return pd.DataFrame()

    @timer(operation_name="IMF.collect_all", track_metrics=True)
    def collect_data(self, indicators: Optional[List[str]] = None) -> pd.DataFrame:
        indicators = indicators or self.config.DEFAULT_IMF_INDICATORS
        self.logger.info(f"💰 Start IMF collection ({len(indicators)} indicators)")
        all_data = []
        with track_progress(indicators, desc="IMF", unit="indicator") as pbar:
            for indicator in pbar:
                pbar.set_postfix_str(f"Indicator: {indicator}")
                df = self._fetch_indicator_data(indicator)
                if not df.empty:
                    all_data.append(df)
                    self.logger.info(f"✅ {len(df)} records collected")
                time.sleep(self.config.DELAY_BETWEEN_REQUESTS)

        result = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
        metric = PerformanceMetrics(
            operation_name="IMF.collect_summary",
            start_time=datetime.now(),
            items_processed=len(result),
        )
        metric.finalize(items=len(result))
        _global_tracker.add_metric(metric)
        self.logger.info(f"✅ IMF done: {len(result)} records")
        return result
#%%
class WHOCollector(AbstractCollector):
    """
    Handles the process of collecting data from the WHO's Global Health Observatory (GHO) API.

    Provides functionality to fetch specific health indicator data, parse it into a structured format,
    and aggregate data for multiple indicators. Ideal for automating the retrieval and handling of WHO's
    health statistics.

    Attributes
    ----------
    config : Config
        The configuration required to access the WHO API, including URLs, default indicators, and
        request delay settings.
    logger : Logger
        Logger instance for recording the progress and any issues during the collection process.

    Methods
    -------
    _parse_who_data(data: Dict, indicator: str) -> List[Dict]
        Parses raw WHO API response data and extracts relevant fields into a structured format.

    _fetch_indicator_data(indicator: str) -> pd.DataFrame
        Fetches data for a specific health indicator from the WHO API, returning it as a DataFrame.

    collect_data(indicators: Optional[List[str]] = None) -> pd.DataFrame
        Collects data for multiple WHO health indicators, aggregates it, and returns the result as
        a unified DataFrame.

    """
    def _parse_who_data(self, data: Dict, indicator: str) -> List[Dict]:
        records = []
        for item in data.get("value", []):
            records.append(
                {
                    "indicator_code": indicator,
                    "indicator_name": item.get("IndicatorCode"),
                    "country_code": item.get("SpatialDim"),
                    "year": pd.to_numeric(item.get("TimeDim"), errors="coerce"),
                    "value": pd.to_numeric(item.get("NumericValue"), errors="coerce"),
                    "source": "WHO GHO",
                }
            )
        return records

    @timer(operation_name="WHO.fetch_indicator", track_metrics=True)
    def _fetch_indicator_data(self, indicator: str) -> pd.DataFrame:
        url = f"{self.config.OMS_API_URL}/{indicator}"
        params = {"$filter": f"SpatialDim eq '{self.config.COUNTRY_CODE}'"}
        response, success = self._make_request_with_retry(url, params=params)
        if not success or response is None:
            return pd.DataFrame()

        try:
            data = response.json()
            records = self._parse_who_data(data, indicator)
            return pd.DataFrame(records)
        except Exception as e:
            self.logger.error(f"❌ Parse error WHO {indicator}: {e}")
            return pd.DataFrame()

    @timer(operation_name="WHO.collect_all", track_metrics=True)
    def collect_data(self, indicators: Optional[List[str]] = None) -> pd.DataFrame:
        indicators = indicators or self.config.DEFAULT_HEALTH_INDICATORS
        self.logger.info(f"🏥 Start WHO collection ({len(indicators)} indicators)")
        all_data = []
        with track_progress(indicators, desc="WHO", unit="indicator") as pbar:
            for indicator in pbar:
                pbar.set_postfix_str(f"Indicator: {indicator}")
                df = self._fetch_indicator_data(indicator)
                if not df.empty:
                    all_data.append(df)
                    self.logger.info(f"✅ {len(df)} records collected")
                time.sleep(self.config.DELAY_BETWEEN_REQUESTS)

        result = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
        metric = PerformanceMetrics(
            operation_name="WHO.collect_summary",
            start_time=datetime.now(),
            items_processed=len(result),
        )
        metric.finalize(items=len(result))
        _global_tracker.add_metric(metric)
        self.logger.info(f"✅ WHO done: {len(result)} records")
        return result
#%%
class UNDPCollector(AbstractCollector):
    """
    Collects and processes data from the UNDP API.

    This class is responsible for retrieving data from the UNDP's API, filtering
    it to include only records relevant to the configured country code, and
    finalizing performance metrics based on the collected data. It is a concrete
    implementation of the AbstractCollector class.

    Methods:
        collect_data: Fetches data from the UNDP API, filters data for a specific
        country based on a configuration, and calculates performance metrics.

    Raises:
        None
    """
    @timer(operation_name="UNDP.collect_all", track_metrics=True)
    def collect_data(self) -> pd.DataFrame:
        self.logger.info("🌐 Start UNDP collection")
        url = self.config.UNDP_API_URL
        response, success = self._make_request_with_retry(url)
        if not success or response is None:
            return pd.DataFrame()

        try:
            from io import BytesIO

            df = pd.read_csv(BytesIO(response.content))
            df_benin = df[df["iso3"] == self.config.COUNTRY_CODE].copy()
            df_benin["source"] = "UNDP HDR"
            metric = PerformanceMetrics(
                operation_name="UNDP.collect_summary",
                start_time=datetime.now(),
                items_processed=len(df_benin),
            )
            metric.finalize(items=len(df_benin))
            _global_tracker.add_metric(metric)
            self.logger.info(f"✅ UNDP: {len(df_benin)} records")
            return df_benin
        except Exception as e:
            self.logger.error(f"❌ UNDP error: {e}")
            return pd.DataFrame()
#%%
class INSAECollector(AbstractCollector):
    """
    Defines the INSAECollector class for collecting and processing data from
    INSAE (Institut National de la Statistique et de l’Analyse Économique). This
    class is responsible for downloading and aggregating data from various Excel
    and CSV files available at specified URLs.

    The class provides methods to scrape and collect data from predefined sources,
    parse the downloaded file contents, handle various exceptions during file
    processing, and consolidate the fetched data into a single Pandas DataFrame.
    """
    @timer(operation_name="INSAE.download_file", track_metrics=True)
    def _download_excel_or_csv(self, url: str) -> pd.DataFrame:
        response, success = self._make_request_with_retry(url)
        if not success or response is None:
            return pd.DataFrame()

        try:
            from io import BytesIO

            if url.endswith(".csv"):
                return pd.read_csv(BytesIO(response.content))
            else:
                return pd.read_excel(BytesIO(response.content), engine="openpyxl")
        except Exception as e:
            self.logger.error(f"❌ File read error: {e}")
            return pd.DataFrame()

    @timer(operation_name="INSAE.scrape_source", track_metrics=True)
    def _scrape_insae_data(self, url: str, source: str) -> pd.DataFrame:
        response, success = self._make_request_with_retry(url)
        if not success or response is None:
            return pd.DataFrame()

        try:
            import re
            from urllib.parse import urljoin

            soup = BeautifulSoup(response.content, "html.parser")
            download_links = soup.find_all("a", href=re.compile(r"\.(xlsx?|csv)$"))
            data_frames = []
            links_to_process = download_links[:5]
            with track_progress(
                    links_to_process, desc=f"INSAE {source}", unit="file"
            ) as pbar:
                for link in pbar:
                    file_url = urljoin(url, link["href"])
                    pbar.set_postfix_str(f"File: {link.get_text(strip=True)[:30]}")
                    self.logger.info(f"📥 Downloading: {file_url}")
                    file_df = self._download_excel_or_csv(file_url)
                    if not file_df.empty:
                        file_df["source"] = f"INSAE - {source}"
                        data_frames.append(file_df)
            return (
                pd.concat(data_frames, ignore_index=True)
                if data_frames
                else pd.DataFrame()
            )
        except Exception as e:
            self.logger.error(f"❌ INSAE scraping error: {e}")
            return pd.DataFrame()

    @timer(operation_name="INSAE.collect_all", track_metrics=True)
    def collect_data(self) -> pd.DataFrame:
        self.logger.info("🇧🇯 Start INSAE Bénin collection")
        insae_urls = self.config.EXTERNAL_SCRAPING_URLS
        all_data = []
        with track_progress(
                insae_urls.items(), desc="INSAE sources", unit="source"
        ) as pbar:
            for source_name, url in pbar:
                if not source_name.startswith(("rgph", "edc", "emicov")):
                    continue
                pbar.set_postfix_str(f"Source: {source_name}")
                df = self._scrape_insae_data(url, source_name)
                if not df.empty:
                    all_data.append(df)

        result = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
        metric = PerformanceMetrics(
            operation_name="INSAE.collect_summary",
            start_time=datetime.now(),
            items_processed=len(result),
        )
        metric.finalize(items=len(result))
        _global_tracker.add_metric(metric)
        self.logger.info(f"✅ INSAE done: {len(result)} records")
        return result
#%%
class DataCleaner:
    """
    Provides a framework for cleaning and preprocessing data.

    The class is designed to standardize and clean datasets across various domains.
    It provides capabilities for removing duplicates, handling null values,
    standardizing column names, converting data types, removing outliers, and more.
    It can also perform specific cleaning operations for World Bank data and
    geographic data. A cleaning summary can be generated to review the results
    of operations performed.

    Attributes:
        config: The configuration dictionary for the cleaning process.
        reports: A dictionary holding the cleaning reports for datasets.

    Methods:
        clean_dataset(df, source_name):
            Cleans the provided dataset with various preprocessing operations.
        clean_world_bank_data(df):
            Cleans World Bank-specific data with rules specific to such datasets.
        clean_geographic_data(df):
            Cleans geographic-specific data, including latitude and longitude validation.
        generate_cleaning_summary():
            Generates a summary DataFrame of all cleaning processes performed.
    """
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config or self._default_config()
        self.reports: Dict[str, CleaningReport] = {}

    @staticmethod
    def _default_config() -> Dict:
        return {
            "remove_duplicates": True,
            "handle_nulls": True,
            "null_threshold": 0.7,
            "detect_outliers": True,
            "outlier_method": "iqr",
            "outlier_threshold": 3,
            "standardize_text": True,
            "standardize_dates": True,
            "convert_types": True,
            "remove_empty_strings": True,
        }

    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        import re

        new_columns = {}
        for col in df.columns:
            new_col = str(col).lower()
            new_col = re.sub(r"[^\w\s]", "", new_col)
            new_col = re.sub(r"\s+", "_", new_col)
            new_col = re.sub(r"_+", "_", new_col).strip("_")
            new_columns[col] = new_col
        df = df.rename(columns=new_columns)
        self.logger.info(f"✅ {len(new_columns)} columns standardized")
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        initial_count = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_count - len(df)
        if duplicates_removed > 0:
            self.logger.info(f"🗑️ {duplicates_removed} duplicates removed")
        return df, duplicates_removed

    def _clean_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        text_cols = df.select_dtypes(include=["object"]).columns
        for col in text_cols:
            df[col] = df[col].str.strip()
            if self.config.get("remove_empty_strings"):
                df[col] = df[col].replace("", np.nan)
                df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)
        return df

    def _convert_data_types(
            self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        conversions = {}
        for col in df.columns:
            if df[col].dtype == "object":
                original_type = str(df[col].dtype)
                numeric_vals = pd.to_numeric(df[col], errors="coerce")
                if numeric_vals.notna().sum() / len(df) > 0.8:
                    df[col] = numeric_vals
                    conversions[col] = f"{original_type} -> numeric"
                    continue
                try:
                    date_vals = pd.to_datetime(df[col], errors="coerce")
                    if date_vals.notna().sum() / len(df) > 0.8:
                        df[col] = date_vals
                        conversions[col] = f"{original_type} -> datetime"
                except:
                    pass
        if conversions:
            self.logger.info(f"🔄 {len(conversions)} type conversions")
        return df, conversions

    def _remove_empty_columns(
            self, df: pd.DataFrame, report: CleaningReport
    ) -> Tuple[pd.DataFrame, List[str]]:
        threshold = self.config.get("null_threshold", 0.7)
        dropped_cols = []
        for col in df.columns:
            null_ratio = df[col].isnull().sum() / len(df)
            if null_ratio > threshold:
                dropped_cols.append(col)
                report.issues_detected.append(
                    f"Column '{col}' dropped: {null_ratio:.1%} missing values"
                )
        if dropped_cols:
            df = df.drop(columns=dropped_cols)
            self.logger.warning(f"⚠️ {len(dropped_cols)} columns dropped")
        return df, dropped_cols

    def _handle_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        if not self.config.get("detect_outliers"):
            return df, 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers_removed = 0
        for col in numeric_cols:
            if df[col].notna().sum() < 10:
                continue
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers_count = outliers_mask.sum()
            if outliers_count > 0:
                df.loc[outliers_mask, col] = np.nan
                outliers_removed += outliers_count
        if outliers_removed > 0:
            self.logger.info(f"🔍 {outliers_removed} outliers handled")
        return df, outliers_removed

    @timer(operation_name="DataCleaner.clean_dataset", track_metrics=True)
    def clean_dataset(
            self, df: pd.DataFrame, source_name: str
    ) -> Tuple[pd.DataFrame, CleaningReport]:
        self.logger.info(f"🧹 Cleaning: {source_name}")
        report = CleaningReport(
            source=source_name, initial_rows=len(df), final_rows=len(df)
        )
        df = self._standardize_column_names(df)
        report.columns_standardized = df.columns.tolist()
        df, duplicates = self._remove_duplicates(df)
        report.duplicates_removed = duplicates
        df = self._clean_text_columns(df)
        df, dropped_cols = self._remove_empty_columns(df, report)
        report.columns_dropped = dropped_cols
        df, conversions = self._convert_data_types(df)
        report.data_types_converted = conversions
        df, outliers = self._handle_outliers(df)
        report.outliers_removed = outliers
        report.final_rows = len(df)
        report.rows_removed = report.initial_rows - report.final_rows
        self.reports[source_name] = report
        self.logger.info(
            f"✅ Cleaning done: {report.initial_rows} → {report.final_rows} rows ({report.rows_removed} removed)"
        )
        return df, report

    def clean_world_bank_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("🌍 World Bank specific cleaning")
        df = df.dropna(subset=["value"])
        df = df[(df["year"] >= 1900) & (df["year"] <= datetime.now().year)]
        df = df[df["value"] >= 0]
        return df

    def clean_geographic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("🗺️ Geographic specific cleaning")
        df = df.dropna(subset=["name"])
        if "latitude" in df.columns and "longitude" in df.columns:
            df = df[
                (df["latitude"].between(-90, 90)) & (df["longitude"].between(-180, 180))
                ]
        return df

    def generate_cleaning_summary(self) -> pd.DataFrame:
        if not self.reports:
            self.logger.warning("⚠️ No cleaning reports available")
            return pd.DataFrame()
        summary_data = [report.to_dict() for report in self.reports.values()]
        summary_df = pd.DataFrame(summary_data)
        self.logger.info("📊 Cleaning summary generated")
        return summary_df
#%%
class DataCollectorOrchestrator:
    """
    Handles the orchestration of data collection, cleaning, and consolidation
    operations by coordinating with different data collectors and processing tools.

    The class is responsible for initializing, organizing, and managing the workflow
    pipeline for collecting raw data from various sources, cleaning it using defined
    cleaning routines, and consolidating final datasets into structured outputs. It
    supports a configurable and extendable workflow, facilitating data operations
    such as storage, tracking progress, and logging.
    """
    def __init__(
        self, config: Optional[GlobalConfig] = None, base_dir: Optional[Path] = None
    ):
        self.config = config or GlobalConfig()
        self.logger = logging.getLogger(__name__)
        self.directory_manager = DirectoryManager(base_dir)
        self.directories = self.directory_manager.initialize_structure_directory()
        self.collectors = {
            "world_bank": WorldBankCollector(self.config),
            "imf": IMFCollector(self.config),
            "who": WHOCollector(self.config),
            "undp": UNDPCollector(self.config),
            "insae": INSAECollector(self.config),
            "web_scraping": WebScrapingCollector(self.config),
            "geographic": GeographicCollector(self.config),
            "external": ExternalCollector(self.config),
        }
        self.cleaner = DataCleaner()
    
    def consolidate_final_data(
        self, cleaned_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        self.logger.info("🔗 Consolidating final datasets")
        final_datasets = {}
        
        economic_sources = ['world_bank', 'imf']
        economic_data = []
        for source in economic_sources:
            if source in cleaned_data and not cleaned_data[source].empty:
                df = cleaned_data[source].copy()
                df['data_source'] = source
                economic_data.append(df)
        
        if economic_data:
            economic_consolidated = pd.concat(economic_data, ignore_index=True)
            final_datasets['economic_indicators'] = economic_consolidated
            filepath = self.directories['final_data'] / 'economic_indicators.csv'
            economic_consolidated.to_csv(filepath, index=False, encoding='utf-8')
            self.logger.info(f"💾 Saved: {filepath.name}")
        
        health_sources = ['who']
        health_data = []
        for source in health_sources:
            if source in cleaned_data and not cleaned_data[source].empty:
                df = cleaned_data[source].copy()
                df['data_source'] = source
                health_data.append(df)
        
        if health_data:
            health_consolidated = pd.concat(health_data, ignore_index=True)
            final_datasets['health_indicators'] = health_consolidated
            filepath = self.directories['final_data'] / 'health_indicators.csv'
            health_consolidated.to_csv(filepath, index=False, encoding='utf-8')
            self.logger.info(f"💾 Saved: {filepath.name}")
        
        geo_sources = [k for k in cleaned_data.keys() if 'geographic' in k]
        for source in geo_sources:
            if source in cleaned_data and not cleaned_data[source].empty:
                df = cleaned_data[source].copy()
                final_datasets[source] = df
                filepath = self.directories['final_data'] / f'{source}.csv'
                df.to_csv(filepath, index=False, encoding='utf-8')
                self.logger.info(f"💾 Saved: {filepath.name}")
        
        other_sources = [k for k in cleaned_data.keys() 
                        if k not in economic_sources + health_sources + geo_sources]
        for source in other_sources:
            if source in cleaned_data and not cleaned_data[source].empty:
                df = cleaned_data[source].copy()
                final_datasets[source] = df
                filepath = self.directories['final_data'] / f'{source}.csv'
                df.to_csv(filepath, index=False, encoding='utf-8')
                self.logger.info(f"💾 Saved: {filepath.name}")
        
        self.logger.info(f"✅ {len(final_datasets)} final datasets created")
        return final_datasets

    @timer(operation_name="Orchestrator.full_collection", track_metrics=True)
    def run_full_collection(
            self, collectors: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        collectors = collectors or list(self.collectors.keys())
        results = {}
        self.logger.info(f"🚀 Starting collection ({len(collectors)} collectors)")
        with track_progress(
                collectors, desc="Global collection", unit="collector"
        ) as pbar:
            for collector_name in pbar:
                if collector_name not in self.collectors:
                    self.logger.warning(f"⚠️ Unknown collector: {collector_name}")
                    continue
                pbar.set_postfix_str(f"Collector: {collector_name}")
                self.logger.info(f"▶️ Starting: {collector_name}")
                try:
                    collector = self.collectors[collector_name]
                    data = collector.collect_data()
                    if collector_name == "geographic" and isinstance(data, dict):
                        for key, df in data.items():
                            if not df.empty:
                                filename = f"{collector_name}_{key}.csv"
                                filepath = self.directories["raw"] / filename
                                collector.save_data(df, filepath)
                                results[f"{collector_name}_{key}"] = df
                        combined = pd.concat(
                            [df for df in data.values() if not df.empty],
                            ignore_index=True,
                        )
                        if not combined.empty:
                            results[collector_name] = combined
                    elif isinstance(data, pd.DataFrame) and not data.empty:
                        results[collector_name] = data
                        filename = f"{collector_name}_data.csv"
                        filepath = self.directories["raw"] / filename
                        collector.save_data(data, filepath)
                    record_count = (
                        len(data)
                        if isinstance(data, pd.DataFrame)
                        else sum(
                            len(df)
                            for df in data.values()
                            if isinstance(df, pd.DataFrame)
                        )
                    )
                    self.logger.info(f"✅ {collector_name}: {record_count} records")
                except Exception as e:
                    self.logger.error(f"❌ Error {collector_name}: {e}", exc_info=True)
        total_records = sum(
            len(df) for df in results.values() if isinstance(df, pd.DataFrame)
        )
        self.logger.info(f"🏁 Collection done: {total_records} records")
        return results

    @timer(operation_name="Orchestrator.full_cleaning", track_metrics=True)
    def run_full_cleaning(
            self, raw_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        self.logger.info(f"🧹 Start cleaning ({len(raw_data)} sources)")
        cleaned_data = {}
        with track_progress(raw_data.items(), desc="Cleaning", unit="source") as pbar:
            for source_name, df in pbar:
                if not isinstance(df, pd.DataFrame) or df.empty:
                    continue
                pbar.set_postfix_str(f"Source: {source_name}")
                try:
                    cleaned_df, report = self.cleaner.clean_dataset(df, source_name)
                    if "world_bank" in source_name.lower():
                        cleaned_df = self.cleaner.clean_world_bank_data(cleaned_df)
                    elif "geographic" in source_name.lower():
                        cleaned_df = self.cleaner.clean_geographic_data(cleaned_df)
                    if not cleaned_df.empty:
                        cleaned_data[source_name] = cleaned_df
                        filepath = (
                                self.directories["processed"] / f"{source_name}_cleaned.csv"
                        )
                        cleaned_df.to_csv(filepath, index=False, encoding="utf-8")
                        self.logger.info(f"💾 Saved: {filepath.name}")
                except Exception as e:
                    self.logger.error(f"❌ Cleaning error {source_name}: {e}")
        self.logger.info(f"✅ Cleaning done: {len(cleaned_data)} sources processed")
        return cleaned_data

    def generate_collection_summary(
            self, data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        summary = []
        for source, df in data.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            date_cols = df.select_dtypes(include=["datetime64"]).columns
            summary.append(
                {
                    "source": source,
                    "records": len(df),
                    "columns": len(df.columns),
                    "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2),
                    "has_nulls": df.isnull().any().any(),
                    "null_pct": round(
                        df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2
                    ),
                    "numeric_cols": len(numeric_cols),
                    "date_cols": len(date_cols),
                    "duplicates": df.duplicated().sum(),
                    "date": datetime.now().date(),
                }
            )
        summary_df = pd.DataFrame(summary)
        if not summary_df.empty:
            filepath = self.directories["processed"] / "collection_summary.csv"
            summary_df.to_csv(filepath, index=False, encoding="utf-8")
            print("\n" + "=" * 100)
            print("📊 COLLECTION SUMMARY")
            print("=" * 100)
            print(summary_df.to_string(index=False))
            print("=" * 100 + "\n")
        return summary_df

    def validate_data_quality(
            self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, List[str]]:
        self.logger.info("🔍 Data quality validation")
        issues = {}
        for source, df in data.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            source_issues = []
            dup_count = df.duplicated().sum()
            if dup_count > 0:
                source_issues.append(f"Duplicates: {dup_count}")
            high_null_cols = df.columns[df.isnull().sum() / len(df) > 0.5]
            if len(high_null_cols) > 0:
                source_issues.append(f"Columns >50% nulls: {list(high_null_cols)}")
            if "year" in df.columns:
                invalid_years = df[(df["year"] < 1900) | (df["year"] > 2025)]
                if len(invalid_years) > 0:
                    source_issues.append(f"Invalid years: {len(invalid_years)}")
            if source_issues:
                issues[source] = source_issues
        if issues:
            self.logger.warning(f"⚠️ Issues detected in {len(issues)} sources")
            for source, issue_list in issues.items():
                for issue in issue_list:
                    self.logger.warning(f"  - {source}: {issue}")
        else:
            self.logger.info("✅ No quality issues detected")
        return issues

    def create_data_dictionary(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        self.logger.info("📖 Creating data dictionary")
        dictionary = []
        for source, df in data.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            for col in df.columns:
                entry = {
                    "source": source,
                    "variable": col,
                    "type": str(df[col].dtype),
                    "non_null_count": df[col].notna().sum(),
                    "null_count": df[col].isnull().sum(),
                    "null_pct": round(df[col].isnull().sum() / len(df) * 100, 2),
                    "unique_values": df[col].nunique(),
                }
                if pd.api.types.is_numeric_dtype(df[col]):
                    entry.update(
                        {
                            "min": df[col].min(),
                            "max": df[col].max(),
                            "mean": round(df[col].mean(), 2),
                            "median": df[col].median(),
                        }
                    )
                dictionary.append(entry)
        dict_df = pd.DataFrame(dictionary)
        if not dict_df.empty:
            filepath = self.directories["docs"] / "data_dictionary.csv"
            dict_df.to_csv(filepath, index=False, encoding="utf-8")
            self.logger.info(f"💾 Dictionary saved: {filepath}")
        return dict_df

    @timer(operation_name="Orchestrator.complete_pipeline", track_metrics=True)
    def run_complete_pipeline(self) -> Dict[str, Any]:
        print("\n" + "=" * 100)
        print("🚀 STARTING COMPLETE ANIP PIPELINE")
        print("=" * 100 + "\n")
        print("📡 STEP 1/4: DATA COLLECTION")
        print("-" * 100)
        raw_data = self.run_full_collection()
        print("\n🔍 STEP 2/4: INITIAL VALIDATION")
        print("-" * 100)
        initial_issues = self.validate_data_quality(raw_data)
        print("\n🧹 STEP 3/4: DATA CLEANING")
        print("-" * 100)
        cleaned_data = self.run_full_cleaning(raw_data)
        print("\n📊 STEP 4/4: GENERATING DELIVERABLES")
        print("-" * 100)
        collection_summary = self.generate_collection_summary(cleaned_data)
        cleaning_summary = self.cleaner.generate_cleaning_summary()
        data_dictionary = self.create_data_dictionary(cleaned_data)
        
        print("\n🔗 STEP 5/5: CONSOLIDATION & EXPORT")
        print("-" * 100)
        final_datasets = self.consolidate_final_data(cleaned_data)
        
        final_issues = self.validate_data_quality(cleaned_data)
        if not cleaning_summary.empty:
            filepath = self.directories["processed"] / "cleaning_summary.csv"
            cleaning_summary.to_csv(filepath, index=False, encoding="utf-8")
            print(f"\n💾 Cleaning summary: {filepath}")
            print("\n" + cleaning_summary.to_string(index=False))
        _global_tracker.print_summary()
        print("\n" + "=" * 100)
        print("✅ PIPELINE COMPLETED")
        print("=" * 100)
        print(f"📂 Raw data: {self.directories['raw']}")
        print(f"📂 Cleaned data: {self.directories['processed']}")
        print(f"📂 Final data: {self.directories['final_data']}")
        print(f"📂 Documentation: {self.directories['docs']}")
        print("=" * 100 + "\n")
        return {
            "raw_data": raw_data,
            "cleaned_data": cleaned_data,
            "final_datasets": final_datasets,
            "collection_summary": collection_summary,
            "cleaning_summary": cleaning_summary,
            "data_dictionary": data_dictionary,
            "initial_issues": initial_issues,
            "final_issues": final_issues,
            "performance_metrics": _global_tracker.get_summary(),
        }
#%%
@timer(operation_name="Main.execution", track_metrics=True)
def main():
    """
    Executes the main data collection, cleaning, and processing pipeline, and provides
    summary results and performance metrics.

    This function serves as the main entry point for orchestrating the entire data workflow.
    It initializes the environment, executes the complete pipeline, logs detailed performance
    metrics, and displays a summary of the processing results. The function tracks the performance
    of operations and maintains a record of any remaining quality issues.

    Parameters:
    operation_name (str): The name of the operation being timed, used for performance tracking.
    track_metrics (bool): Whether to track and log performance metrics. Defaults to True.

    Returns:
    dict: A dictionary containing the results of the pipeline execution. Keys include:
        - 'raw_data': List of collected raw data sources.
        - 'cleaned_data': List of cleaned data sources.
        - 'final_datasets': List of finalized datasets.
        - 'data_dictionary': List of variables documented in the datasets.
        - 'final_issues': List of sources with unresolved quality issues.
        - 'performance_metrics': Dictionary with performance metrics, such as total duration,
          number of successful and failed operations, and items processed.

    Raises:
    Exception: If an error occurs during any phase of the data workflow.

    Notes:
    The function logs detailed performance metrics to a CSV file within the designated log
    directory. The performance data includes operation-specific metrics such as execution
    time and success rates.

    See Also:
    DataCollectorOrchestrator: Class responsible for orchestrating data collection and cleaning.
    setup_environment: Initializes the necessary environment and logging setup.
    _global_tracker: Tracks operational metrics for performance evaluations.
    """
    setup_environment(log_dir=Path("logs"))
    orchestrator = DataCollectorOrchestrator()
    results = orchestrator.run_complete_pipeline()
    print("\n📋 FINAL RESULTS:")
    print(f"  - Sources collected: {len(results['raw_data'])}")
    print(f"  - Sources cleaned: {len(results['cleaned_data'])}")
    print(f"  - Final datasets: {len(results['final_datasets'])}")
    print(f"  - Variables documented: {len(results['data_dictionary'])}")
    if results["final_issues"]:
        print(f"  ⚠️ Remaining issues: {len(results['final_issues'])} sources")
    else:
        print("  ✅ No quality issues detected")
    print("\n⏱️ GLOBAL PERFORMANCE:")
    perf = results["performance_metrics"]
    print(f"  - Total operations: {perf['total_operations']}")
    print(f"  - Successful: {perf['successful']}")
    print(f"  - Failed: {perf['failed']}")
    print(f"  - Total duration: {perf['total_duration']}s")
    print(f"  - Average duration: {perf['avg_duration']}s")
    print(f"  - Items processed: {perf['total_items']}")
    performance_df = pd.DataFrame([m.to_dict() for m in _global_tracker.metrics])
    perf_filepath = orchestrator.directories["logs"] / "performance_metrics.csv"
    performance_df.to_csv(perf_filepath, index=False, encoding="utf-8")
    print(f"\n💾 Performance metrics: {perf_filepath}")
    return results
#%%
_ = main()