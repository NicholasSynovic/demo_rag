import pickle
from os.path import exists
from pathlib import Path
from typing import List

import chromadb
import torch
from chromadb.utils import embedding_functions
from datasets import DatasetDict, load_dataset
from numpy import ndarray
from progress.bar import Bar
from sentence_transformers import SentenceTransformer
from torch import Tensor


def getDataset(
    name: str = "fancyzhx/ag_news",
    stor: Path = Path("./data"),
) -> DatasetDict:
    ds: DatasetDict = load_dataset(path=name, cache_dir=stor)
    return ds


def getArticles(ds: DatasetDict) -> List[str]:
    return ds["train"]["text"]


def embedDocuments(
    documents: List[str],
    model: str = "all-MiniLM-L6-v2",
    outputFP: Path = Path(
        "embeddings.pickle",
    ),
) -> ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st: SentenceTransformer = SentenceTransformer(
        model_name_or_path=model,
        model_kwargs={"torch_dtype": "float16"},
        device=device,
    )

    embeddings: ndarray = st.encode(sentences=documents, show_progress_bar=True)

    with open(outputFP, mode="wb") as pf:
        pickle.dump(obj=embeddings, file=pf)
        pf.close()

    return embeddings


def loadEmbeddings(embeddings: ndarray, dbPath: Path = Path("embeddings.db")) -> None:
    client = chromadb.PersistentClient(path=dbPath.__str__())
    collection = client.get_or_create_collection(name="embeddings")

    with Bar("Loading embeddings into DB...", max=len(embeddings)) as bar:
        for idx in range(len(embeddings)):
            collection.add(ids=str(idx), embeddings=embeddings[idx])
            bar.next()


if __name__ == "__main__":
    pfp: Path = Path("./embeddings.pickle")

    if not exists(path=pfp):
        ds: DatasetDict = getDataset()
        articles: List[str] = getArticles(ds=ds)
        embeddings = embedDocuments(documents=articles, outputFP=pfp)

    embeddings = pickle.load(file=open("embeddings.pickle", "rb"))

    loadEmbeddings(embeddings=embeddings)
