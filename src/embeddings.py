import pickle  # nosec
from os.path import exists
from pathlib import Path
from typing import List

import chromadb
import torch
from chromadb import Collection
from chromadb.api import ClientAPI
from datasets import DatasetDict, load_dataset
from numpy import ndarray
from progress.bar import Bar
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
        model_name_or_path=embeddingModel,
        model_kwargs=model_kwargs,
        cache_folder=modelDir,
        device=computeDevice,
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


def storeDocumentEmbeddings(
    documents: List[str],
    embeddings: ndarray,
    dbPath: Path = Path("./db"),
) -> None:
    if len(documents) != len(embeddings):
        raise IndexError(
            "`documents` parameter length != `embeddings` parameter length",
        )

    dbClient: ClientAPI = chromadb.PersistentClient(path=dbPath.__str__())
    collection: Collection = dbClient.get_or_create_collection(
        name="embeddings",
    )

    with Bar(
        "Adding embeddings and documents to ChromaDB database...",
        max=len(documents),
    ) as bar:
        idx: int
        for idx in range(len(documents)):
            _id: str = str(idx)
            document: str = documents[idx]
            embedding: ndarray = embeddings[idx]

            collection.add(ids=_id, embeddings=embedding, documents=document)
            bar.next()


def main() -> None:
    ds: DatasetDict = downloadDataset()
    print("Downloaded dataset")

    documents: List[str] = getDocumentsFromDataset(dataset=ds)
    print("Got dataset documents")

    embeddings: ndarray
    if exists(path=Path("./data/embeddings.pickle")):
        embeddings = pickle.load(  # nosec
            file=open(file=Path("./data/embeddings.pickle"), mode="rb"),
        )

    else:
        embeddings = createDocumentEmbeddings(documents=documents)

    print("Loaded embeddings")

    storeDocumentEmbeddings(documents=documents, embeddings=embeddings)
    print("Stored documents and embeddings in ChromaDB")


if __name__ == "__main__":
    main()
