#from llama_index.embeddings.huggingface import HuggingFaceEmbedding
#from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.groq import Groq
from llama_index.llms.ollama import Ollama
import os
from dotenv import load_dotenv
import qdrant_client


def get_env_from_colab_or_os(key):
    try:
        from google.colab import userdata

        try:
            return userdata.get(key)
        except userdata.SecretNotFoundError:
            pass
    except ImportError:
        pass
    return os.getenv(key)


load_dotenv()

EMBED_MODEL = OllamaEmbedding(
    model_name="bge-m3:latest",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

client = qdrant_client.QdrantClient(
    # you can use :memory: mode for fast and light-weight experiments,
    # it does not require to have Qdrant deployed anywhere
    # but requires qdrant-client >= 1.1.1
    # location=":memory:"
    # otherwise set Qdrant instance address with:
    # url="http://<host>:<port>"
    # otherwise set Qdrant instance with host and port:
    host="localhost",
    port=6333
    # set API KEY for Qdrant Cloud
    # api_key="<qdrant-api-key>",
)

GEN_MODEL = Groq(model="llama3-groq-8b-8192-tool-use-preview", api_key=os.getenv("GROQ_API_KEY"))
#GEN_MODEL = Ollama(model="aya-expanse:8b-q8_0", base_url='http://localhost:11434')
#GEN_MODEL = Ollama(model="llama3.2:3b-instruct-q8_0", base_url='http://localhost:11434')

SOURCE = "e://Work//RAG//Advanced-Hybrid-RAG-with-BM25-and-Llama-3.1//src//temp//2409.13740v2.pdf"  # Docling Technical Report
#QUERY = "How to evaluate PaperQA2 on summarization?"

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.readers.docling import DoclingReader
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine

from fastembed import SparseTextEmbedding
import qdrant_client
from qdrant_client import QdrantClient, models


from transformers import AutoTokenizer

from docling.chunking import HybridChunker


EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 1024

tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
chunker = HybridChunker(
        tokenizer=tokenizer,  # can also just pass model name instead of tokenizer instance
        max_tokens=MAX_TOKENS,  # optional, by default derived from `tokenizer`
        # merge_peers=True,  # optional, defaults to True
)

reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
node_parser = DoclingNodeParser(chunker=chunker)

#reader = DoclingReader(export_type=DoclingReader.ExportType.MARKDOWN)
#node_parser = MarkdownNodeParser()

ollama_embedding = OllamaEmbedding(
    model_name="bge-m3:latest",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

client = qdrant_client.QdrantClient(
    # you can use :memory: mode for fast and light-weight experiments,
    # it does not require to have Qdrant deployed anywhere
    # but requires qdrant-client >= 1.1.1
    # location=":memory:"
    # otherwise set Qdrant instance address with:
    # url="http://<host>:<port>"
    # otherwise set Qdrant instance with host and port:
    host="localhost",
    port=6333
    # set API KEY for Qdrant Cloud
    # api_key="<qdrant-api-key>",
)

def create_sparse_vector(query_list):
    """
        Create a sparse vector from the text using BM25.
    """
    model = SparseTextEmbedding(model_name="Qdrant/bm25")

    indices_list = []
    values_list  = []
    for text in query_list:
        embeddings = list(model.embed(text))[0]

        #sparse_vector = models.SparseVector(
        indices=embeddings.indices.tolist()
        values=embeddings.values.tolist()

        indices_list.append(indices)
        values_list.append(values)

    return indices_list, values_list


vector_store = QdrantVectorStore(client=client, 
                                 collection_name="cmos_vlsi_4ed",
                                 #collection_name="collection_bm25_256_0",
                                 #fastembed_sparse_model="Qdrant/bm25",
                                 sparse_doc_fn=create_sparse_vector,
                                 sparse_query_fn=create_sparse_vector,
                                 #hybrid_fusion_fn="rrf",
                                 enable_hybrid=True)
#try:
#    vector_store.clear()
#except:
#    pass

storage_context = StorageContext.from_defaults(vector_store=vector_store)

#index = VectorStoreIndex.from_documents(
#    documents=reader.load_data("e://Work//RAG//Advanced-Hybrid-RAG-with-BM25-and-Llama-3.1//src//temp//CMOS_VLSI_Design_A_Circuits_and_Systems_Perspective_4th_Edition.pdf"),
#    documents=reader.load_data("e://Work//RAG//Advanced-Hybrid-RAG-with-BM25-and-Llama-3.1//src//temp//2409.13740v2.pdf"),
#    transformations=[node_parser],
#    embed_model=ollama_embedding,
#    storage_context=storage_context,
#    batch_size=20,
#)

index = VectorStoreIndex.from_vector_store(
    embed_model=ollama_embedding,
    vector_store=vector_store
)

from llama_index.core import PromptTemplate

qa_prompt_tmpl = (
    "Context information is below.\n"
    "-------------------------------"
    "{context_str}\n"
    "-------------------------------"
    "Given the context information and not prior knowledge,"
    "answer the query. Please be concise, and complete.\n"
    "If the context does not contain an answer to the query,"
    "respond with \"Sorry, I don't know!\"."
    "Query: {query_str}\n"
    "Answer: "
)
qa_prompt = PromptTemplate(qa_prompt_tmpl)

Settings.embed_model = EMBED_MODEL
Settings.llm = GEN_MODEL

# retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=3,
    sparse_top_k=3,
    vector_store_query_mode="hybrid"
)

# response synthesizer
response_synthesizer = get_response_synthesizer(
    llm=GEN_MODEL,
    text_qa_template=qa_prompt,
    response_mode="compact",
)

from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.chat_engine.types import ChatMode


memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

chat_engine = index.as_chat_engine(
    memory=memory,
    chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT, 
    llm=GEN_MODEL, 
    verbose=True,
)

chat_history=[ChatMessage.from_str(role='user', content='What\'s the definition of setup time for a d-type flip-flop?'),
              ChatMessage.from_str(role='assistant', content='The setup time for a D-type flip-flop is defined as the smallest value of tDC such that tCQ ≤ tPCQ.  It is the time that the data must set up before the clock edge to ensure proper capture.'),
              ChatMessage.from_str(role='user', content='What\'s the definition of hold time for a d-type flip-flop?'),
              ChatMessage.from_str(role='assistant', content='The hold time for a D-type flip-flop is the minimum time difference between the clock\'s rising edge and the data input\'s valid time. It ensures that the data input remains stable for a certain period after the clock\'s rising edge to prevent unwanted changes in the output.'),
              ChatMessage.from_str(role='user', content='What\'s the difference between them?'),
              ChatMessage.from_str(role='assistant', content='TThe setup time ensures that the data is stable before the clock edge, while the hold time ensures that the data remains stable after the clock edge. Both are critical for maintaining the integrity of the flip-flop\'s output.'),
                                 
            ]
result = chat_engine.chat(
                  #message = 'Thank. And also tell me what the antenna effect is in IC layout? I mean the effect related to the \'plasma-induced gate-oxide damage\'',
                  #message = 'Thank. And also tell me what the antenna effect is in IC layout?', 
                  #message = 'Thank. And also tell who the current US president is?',  
                  #message='What\'s the difference between them?', 
                  message='Are both characteristics interrelated?', 
                  chat_history=chat_history
                )

print(result)
#print(result.sources)
#print(result.response)


# query engine
#query_engine = RetrieverQueryEngine(
#    retriever=retriever,
#    response_synthesizer=response_synthesizer,
#)

#result = index.as_query_engine(llm=GEN_MODEL).query(QUERY)
#QUERIES = ["What's the definition of setup time for a d-type flip-flop?",
#        "What's the definition of hold time for a d-type flip-flop?",
#        "What's the definition of What's the antenna effect in IC layout?",
#        "Who's the current US president?"
#]

#for query in QUERIES:
#    result = query_engine.query(query)
#    #result = index.as_query_engine(llm=GEN_MODEL).query(query)
#    print(f"Q: {query}\nA: {result.response.strip()}\n\nSources:")
#    print([(n.text, n.metadata) for n in result.source_nodes])
#    print("\n\n")