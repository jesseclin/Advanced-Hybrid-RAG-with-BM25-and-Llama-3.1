from ollama import Client
from rag_evaluator import evaluate_rag_relevance  # Assuming previous code is saved as evaluate_rag.py
from rerankers import Reranker

client = Client(
  host='http://localhost:11434',
)

class generate:
    def __init__(self):
        self.question=""
        self.context=""
        self.prompt=""
    def llm_query(self,question,context)->str:
        self.question=question
        self.context=context
        self.prompt=f"""Your task is to provide a clear, concise, and informative explanation based on the following context and query.

        Context:
        {context}

        Query: {question}

        Please follow these guidelines in your response:
        1. Start with a brief overview of the concept mentioned in the query.
        2. Dont mention like answer to your question or such things just the answer is enough
        3. Answer should be in 200-300 words and make it as paras if required.
        Your explanation should be informative yet accessible, suitable for someone with a basic understanding of RAG. If the query asks for information not present in the context, please state that you don't have enough information to provide a complete answer, and only respond based on the given context.
        """
        chat_title=client.chat(messages=[{
              "role":"user",
              "content": self.prompt
          }],model="llama3.2:3b-instruct-q8_0")
        return chat_title.message.content

if __name__ == '__main__':
    from retriever import retriver
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-c', '--collection', help='Qdrant\'s collection name')  
    args = parser.parse_args() 
    collection_name = args.collection if args.collection is not None else "collection_bm25"
    print(collection_name)

    search = retriver(collection_name=collection_name)
    #query = "請問何謂 VLSI 設計中所提到的 \"antenna effect\"?"
    #query = "What's the difference between PLL and DLL? Please explain in Taiwan\'s traditional Chinese."
    #query = "What's the meaning of MTBF? Please explain in Taiwan\'s traditional Chinese."
    #query = "What's the meaning of setup time of a D-type flip-flop?"
    #query = "setup time of a D-type flip-flop"
    #query = "Is the setup time for a D-type flip-flop related to it's the hold-time ?"
    #query = "How to measure the setup time for a D-type flip-flop?"
    #query = "What's the meaning of hold time of a D-type flip-flop? Please explain in Taiwan\'s traditional Chinese."
    #query = "What's the difference between the setup time and the hold time of a d-type flip-flop?"
    #query = "FIGURE 1.4 Transistors in Intel microprocessors [Intel10]"
    query = "FIGURE 1.24 Inefficient discrete gate implementation of AOI22 with transistor counts indicated"
    #query = "FIGURE 1.11 Inverter schematic (a) and symbol (b) Y = A"
    retrieved_docs = search.hybrid_search(query)
    retrievals = [retrieved_docs[idx]['text'] for idx in range(len(retrieved_docs))]
    scores = evaluate_rag_relevance(query, retrievals)
    #context = "\n\n".join([str({'headings': retrieved_docs[idx]['headings'],'text':retrieved_docs[idx]['text']}) for idx in range(len(retrieved_docs))])
    #context = "\n\n".join([retrieved_docs[idx]['text'] for idx in range(len(retrieved_docs))])
    #print(scores)
    #print(f"{context}\n--------------------\n")
    ranker = Reranker('flashrank')
    results = ranker.rank(query=query, docs=retrievals)
    print(results.top_k(1)[0].text)

    score_max = 0
    context = ""
    count = 0
    for idx, content in enumerate(retrievals):
        print(f"Score: {scores[idx]}")
        print("====================\n")
        print(f"Content: {content}")
        print("--------------------")
        print("--------------------\n")
        if score_max < scores[idx]:
            context = content
            count = 1
            score_max = scores[idx]
        else:
            context += f" {content}"
            count += 1
    
    
    search = generate()
    print(count)
    context = results.top_k(1)[0].text
    results = search.llm_query(query, context)
    print(results)
