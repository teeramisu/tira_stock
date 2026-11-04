"""Dataset version helpers for reproducible market data reads."""

from __future__ import annotations

from datetime import datetime

from .errors import UnstableDatasetVersion
from .models import DatasetVersion
from .repository import MarketDataRepository


def build_dataset_version_id(source: str, markets: tuple[str, ...], created_at: datetime | None = None) -> str:
    """Build a deterministic-looking dataset version id.

    Args:
        source: Source label.
        markets: Covered markets.
        created_at: Optional timestamp. Current UTC time is used when omitted.

    Returns:
        Dataset version id.
    """
    ts = (created_at or datetime.utcnow()).strftime("%Y%m%d%H%M%S")
    return f"{source}-{'-'.join(markets)}-{ts}"


def require_stable_dataset_version(
    repository: MarketDataRepository,
    dataset_version: str,
) -> DatasetVersion:
    """Load and validate a stable dataset version.

    Args:
        repository: Market data repository.
        dataset_version: Dataset version id.

    Returns:
        Stable dataset version metadata.

    Raises:
        UnstableDatasetVersion: If the version exists but is not stable.
    """
    version = repository.get_dataset_version(dataset_version)
    if not version.is_stable:
        raise UnstableDatasetVersion(f"Dataset version is not stable: {dataset_version}")
    return version

