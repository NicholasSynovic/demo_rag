import pickle  # nosec
from pathlib import Path
from typing import List

import torch
from datasets import DatasetDict, load_dataset
from numpy import ndarray
from sentence_transformers import SentenceTransformer


def downloadDataset(
    hfRepo: str = "fancyzhx/ag_news",
    datasetDir: Path = Path("./data"),
) -> DatasetDict:
    return load_dataset(
        path=hfRepo,
        cache_dir=datasetDir,
    )


def getDocumentsFromDataset(dataset: DatasetDict) -> List[str]:
    return dataset["train"]["text"]


def createDocumentEmbeddings(
    documents: List[str],
    embeddingModel: str = "all-MiniLM-L6-v2",
    modelDir: Path = Path("./model"),
    embeddingPickleFP: Path = Path("./data/embeddings.pickle"),
    model_kwargs: dict = {"torch_dtype": "float16"},
) -> ndarray:
    computeDevice: str = "cuda" if torch.cuda.is_available() else "cpu"

    st: SentenceTransformer = SentenceTransformer(
        model_name_or_path="all-MiniLM-L6-v2",
        device=computeDevice,
        cache_folder=modelDir,
        backend="torch",
    )

    embeddings: ndarray = st.encode(
        sentences=documents,
        show_progress_bar=True,
    )

    pickle.dump(
        obj=embeddings,
        file=open(
            file=embeddingPickleFP,
            mode="wb",
        ),
    )

    return embeddings


def storeDocumentEmbeddings():
    pass
