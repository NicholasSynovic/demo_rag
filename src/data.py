from pathlib import Path
from typing import List

from datasets import DatasetDict, load_dataset


def getDataset(
    name: str = "fancyzhx/ag_news",
    stor: Path = Path("./data"),
) -> DatasetDict:
    ds: DatasetDict = load_dataset(path=name, cache_dir=stor)
    return ds


def getArticles(ds: DatasetDict) -> List[str]:
    return ds["train"]["text"]
