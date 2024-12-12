import torch
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient, models
from PyPDF2 import PdfReader
import logging
from transformers import AutoTokenizer, AutoModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from typing import List
import uuid
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from fastembed import SparseTextEmbedding

from typing import Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument

from docling.document_converter import DocumentConverter
from docling.datamodel.document import ConversionResult
from docling_core.transforms.chunker import HierarchicalChunker

class DoclingPDFLoader(BaseLoader):

    def __init__(self, file_path: str | list[str]) -> None:
        self._file_paths = file_path if isinstance(file_path, list) else [file_path]
        self._converter = DocumentConverter()

    #def lazy_load(self) -> Iterator[LCDocument]:
    #    for source in self._file_paths:
    #        dl_doc = self._converter.convert(source).document
    #        text = dl_doc.export_to_markdown()
    #        yield LCDocument(page_content=text)
    def lazy_load(self) -> Iterator[ConversionResult]:
        for source in self._file_paths:
            dl_doc = self._converter.convert(source)
            yield dl_doc

class QdrantIndexing:
    def __init__(self, pdf_path):
        """
        Initialize the QdrantIndexing object.
        """
        self.pdf_path = pdf_path
        self.qdrant_client = QdrantClient(url="http://localhost:6333",timeout=100)
        self.collection_name = "collection_bm25"
        self.document_text = ""
        self.bm25 = None
        self.vectorizer = CountVectorizer(binary=True)
        self.model = None
        logging.info("QdrantIndexing object initialized.")

    def read_pdf(self):
        """
        Read text from the PDF file.
        """
        try:
            loader = DoclingPDFLoader(file_path=self.pdf_path)
            self.document_text = loader.load()

            #reader = PdfReader(self.pdf_path)
            #text = ""
            #for page in reader.pages:
            #    text += page.extract_text()  # Extract text from each page
            #self.document_text = text
            logging.info(f"Extracted text from PDF: {self.pdf_path}")
        except Exception as e:
            logging.error(f"Error reading PDF: {e}")

    def client_collection(self):
        """
        Create a collection in Qdrant vector database.
        """
        if self.qdrant_client.collection_exists("collection_bm25"):
            self.qdrant_client.delete_collection("collection_bm25")
        if not self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                     'dense': models.VectorParams(
                         size=384,
                         distance=models.Distance.COSINE,
                     )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                              index=models.SparseIndexParams(
                                on_disk=False,              
                            ),
                        )
                    }
            )
            logging.info(f"Created collection '{self.collection_name}' in Qdrant vector database.")

    def chunk_text(self, docs: LCDocument) -> List[str]:
        """
        Split the text into overlapping chunks.
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        return chunks

    def initialize_bm25(self):
        """
        Initialize BM25 with the document chunks.
        """
        self.model = SparseTextEmbedding(model_name="Qdrant/bm25")

    def create_sparse_vector(self, text):
        """
        Create a sparse vector from the text using BM25.
        """
        embeddings = list(self.model.embed(text))[0]

        sparse_vector = models.SparseVector(
            indices=embeddings.indices.tolist(),
            values=embeddings.values.tolist()
        )
        return sparse_vector

    def get_dense_embedding(self, text):
        """
        Get dense embedding for the given text using BERT-based model.
        """
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embedding = model.encode(text)
        return embedding.tolist()


    def document_insertion(self):
        """
        Insert the document text along with its dense and sparse vectors into Qdrant.
        """
        #chunks = self.chunk_text(self.document_text)
        docs = self.document_text
        chunks = []
        for idx, doc in enumerate(docs):
            chunks += HierarchicalChunker().chunk(docs[idx].document)
        #print(chunks)
        self.initialize_bm25()
        for chunk_index, chunk in enumerate(chunks):
        #for chunk_index, chunk in enumerate(HierarchicalChunker().chunk(docs[0].document)):
            dense_embedding = self.get_dense_embedding(chunk.text)
            sparse_vector = self.create_sparse_vector(chunk.text)
            chunk_id = str(uuid.uuid4())
            chunk_meta = chunk.meta.export_json_dict()
            chunk_headings = ''
            if 'headings' in chunk_meta.keys():
                chunk_headings = chunk_meta['headings']
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    {
                        "id": chunk_id,
                        "vector": {
                            'dense': dense_embedding,
                            'sparse': sparse_vector,
                        },
                        "payload": {
                            'chunk_index': chunk_index,
                            'text': chunk.text,
                            #'metadata': chunk.meta.export_json_dict(),
                            'page_no': chunk_meta['doc_items'][0]['prov'][0]['page_no'],
                            'filename': chunk_meta['origin']['filename'],
                            'headings': chunk_headings,
                            #'captions': chunk.meta.captions,
                        }
                    }]
            )
            if chunk_index<10:
                print(chunk_meta.keys())
            logging.info(f"Inserted chunk {chunk_index + 1}/{len(chunks)} into Qdrant.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    pdf_file_path = "data/Module-1.pdf"
    indexing = QdrantIndexing(pdf_path=pdf_file_path)
    indexing.read_pdf()
    indexing.client_collection()
    indexing.document_insertion()