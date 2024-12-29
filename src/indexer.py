import re
import sys
import torch
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient, models
import logging
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from typing import List
import uuid
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from fastembed import SparseTextEmbedding

from typing import Iterator
from langchain_core.document_loaders import BaseLoader

from docling.document_converter import DocumentConverter
from docling.datamodel.document import ConversionResult
from docling_core.transforms.chunker import HierarchicalChunker

from transformers import AutoTokenizer

from docling.chunking import HybridChunker

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 256

from ollama import Client

oclient = Client(
  host='http://localhost:11434',
)

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
    def __init__(self, pdf_path, collection_name="collection_bm25"):
        """
        Initialize the QdrantIndexing object.
        """
        self.pdf_path = pdf_path
        self.qdrant_client = QdrantClient(url="http://localhost:6333",timeout=100)
        self.collection_name = collection_name
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

    def client_collection(self, recreate=True):
        """
        Create a collection in Qdrant vector database.
        """
        if self.qdrant_client.collection_exists(self.collection_name) and recreate==True:
            self.qdrant_client.delete_collection(self.collection_name)
            print("[INFO] Delete collection{self.collection_name}")
        if not self.qdrant_client.collection_exists(self.collection_name):
            emb_sz = len(self.get_dense_embedding("test"))
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                     'text-dense': models.VectorParams(
                         size=emb_sz,#1024,#384,
                         distance=models.Distance.COSINE,
                     )
                },
                sparse_vectors_config={
                    "text-sparse": models.SparseVectorParams(
                              index=models.SparseIndexParams(
                                on_disk=False,              
                            ),
                        )
                    }
            )
            logging.info(f"Created collection '{self.collection_name}' in Qdrant vector database.")

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
        #model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        #embedding = model.encode(text).tolist()
        #embedding = oclient.embeddings(model="all-minilm:33m", prompt=text)["embedding"]
        embedding = oclient.embeddings(model="bge-m3:latest", prompt=text)["embedding"]
        return embedding

    def strip_figure_captions(self, text):
        """
        Strips out figure captions and their surrounding empty lines from text that match the pattern:
        FIGURE X.X followed by text starting with a capital letter or parenthesis
    
        Args:
            text (str): Input text containing figure captions
        
        Returns:
            str: Text with figure captions and surrounding empty lines removed
        """
        patterns = [r'^\s*FIGURE\s+\d+\.\d+\s+[A-Z(].*$',   # Caption
                    r'^\s*\d{1,4}\s*$',                     # Page number
                    r'^\s*Chapter\s+\d+\s+[A-Z].*$',        # Chapter 
                    r'^\s*[\d\.]+\s+[A-Z].*$',              # Section
                    ]
        lines = text.split('\n')
    
        # Remove figure captions and consolidate empty lines
        filtered_lines = []
        skip_next_empty = False
    
        for i, line in enumerate(lines):
            # Check if current line is a figure caption
            for pattern in patterns:
                is_figure = bool(re.match(pattern, line))
                if is_figure:
                    break
        
            # Check if current line is empty
            is_empty = not line.strip()
        
            # Skip figure captions and manage empty lines
            if is_figure:
                skip_next_empty = True
                continue
        
            # Skip empty line after figure caption
            if is_empty and skip_next_empty:
                skip_next_empty = False
                continue
            
            filtered_lines.append(line)
    
        # Remove consecutive empty lines
        result = []
        prev_empty = False
        for line in filtered_lines:
            is_empty = not line.strip()
            if not (is_empty and prev_empty):
                result.append(line)
            prev_empty = is_empty
            
        return '\n'.join(result)

    def document_insertion(self):
        """
        Insert the document text along with its dense and sparse vectors into Qdrant.
        """
        tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)

        chunker = HybridChunker(
            tokenizer=tokenizer,  # can also just pass model name instead of tokenizer instance
            max_tokens=MAX_TOKENS,  # optional, by default derived from `tokenizer`
            # merge_peers=True,  # optional, defaults to True
        )

        #chunks = self.chunk_text(self.document_text)
        docs = self.document_text
        chunks = []
        for _, doc in enumerate(docs):
            #chunks += HierarchicalChunker().chunk(doc.document)
            chunks += chunker.chunk(dl_doc=doc.document)
        #print(chunks)
        self.initialize_bm25()
        label_cnt = {}
        filename = ""
        eff_chunks = []
        for chunk_index, chunk in enumerate(chunks):
        #for chunk_index, chunk in enumerate(HierarchicalChunker().chunk(docs[0].document)):
            
            chunk_id = str(uuid.uuid4())
            chunk_meta = chunk.meta.export_json_dict()
            chunk_headings = ''
            if 'headings' in chunk_meta.keys():
                chunk_headings = chunk_meta['headings']
            label = chunk_meta['doc_items'][0]['label']
            if label in label_cnt.keys():
                label_cnt[label] += 1
            else:
                label_cnt[label] = 0

            if label in ['text'] and chunk_headings not in ['References','REFERENCES']:
                fixed_text = self.strip_figure_captions(chunk.text)
            else:
                fixed_text = chunk.text
            dense_embedding = self.get_dense_embedding(fixed_text)
            sparse_vector = self.create_sparse_vector(fixed_text)
            if len(dense_embedding)!=1024:
                print(label)
                print(chunk.text)
                print(fixed_text)
                #sys.exit()
                continue
            #if label in ['page_header', 'page_footer', 'caption'] :
            #if label in['paragraph']:
            #    print(f"[{chunk_meta}]{chunk.text}")
            #if label in ['text', 'caption', 'list_item', 'paragraph', 'footnote']:
            #if label in ['text', 'caption'] and 'origin' in chunk_meta.keys() and chunk_headings not in ['References','REFERENCES']:
            if 'origin' in chunk_meta.keys():
                filename = chunk_meta['origin']['filename']
            #else:
            #    print(f"[{chunk_index}] {chunk.text}")
            #    print(chunk_meta)
            if label in ['text'] and chunk_headings not in ['References','REFERENCES']:
                eff_chunks.append({'dense_embedding':dense_embedding, 
                                   'sparse_vector': sparse_vector,
                                   'chunk_id': chunk_id,
                                   'text': fixed_text,
                                   'page_no': chunk_meta['doc_items'][0]['prov'][0]['page_no'],
                                   'filename': filename,
                                   'headings': chunk_headings,
                                   })
                
        for chunk_index, chunk in enumerate(eff_chunks):
            context = ""
            start   = 0
            end     = 1
            current_heading = chunk['headings']
            i=1
            while True:
                if chunk_index-i >=0:
                    if eff_chunks[chunk_index-i]['headings'] == current_heading:
                        start -= 1
                        i += 1
                    else:
                        break
                else:
                    break
            i=1
            while True:
                if chunk_index+i < len(eff_chunks):
                    if eff_chunks[chunk_index+i]['headings'] == current_heading:
                        end += 1
                        i += 1
                    else:
                        break
                else:
                    break    
            #if chunk_index==0:
            #    start = 0
            #elif chunk_index==len(eff_chunks)-1:
            #    end   = 0
            print(f"[{chunk_index}] {start}->{end}")

            for i in range(start, end, 1):
                #print(f"1:{current_heading}")
                #print(f"2:{eff_chunks[chunk_index+i]['headings']}")
                #if eff_chunks[chunk_index+i]['headings'] == current_heading:
                context += f"[{i}]{eff_chunks[chunk_index+i]['text']}\n\n"
            #print(context)        
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    {
                        "id": chunk['chunk_id'],
                        "vector": {
                            'text-dense': chunk['dense_embedding'],
                            'text-sparse': chunk['sparse_vector'],
                        },
                        "payload": {
                            'chunk_index': chunk_index,
                            'text': context,
                            #'metadata': chunk.meta.export_json_dict(),
                            'page_no' : chunk['page_no'],
                            'filename': chunk['filename'],
                            'headings': chunk['headings'],
                            #'captions': chunk.meta.captions,
                        }
                    }
                ]
            )
                #if chunk_index>100:
                #    print(f"[{chunk_index}] {chunk.text}")
                #    print(chunk_meta)
            logging.info(f"Inserted chunk {chunk_index + 1}/{len(eff_chunks)} into Qdrant.")

        print(label_cnt.keys())

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    logging.basicConfig(level=logging.INFO)
    
    parser.add_argument('file', help='Input PDF File')
    parser.add_argument('-c', '--collection', help='Qdrant\'s collection name')
    parser.add_argument('-r', '--recreate', action='store_true', help='Recrate Qdrant\'s collection')  
    args = parser.parse_args() 

    pdf_file_path = args.file
    collection_name = args.collection if args.collection is not None else "collection_bm25"
    recrate = args.recreate
    print(f"[{recrate}]")
    
    indexing = QdrantIndexing(pdf_path=pdf_file_path, collection_name=collection_name)
    indexing.read_pdf()
    indexing.client_collection(recreate=recrate)
    indexing.document_insertion()