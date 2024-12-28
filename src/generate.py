from ollama import Client

client = Client(
  host='http://192.168.55.206:11435',
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
          }],model="gemma2:27b-instruct-q8_0")
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
    query = "FIGURE 11.10"
    retrieved_docs = search.hybrid_search(query)
    #context = "\n\n".join([str({'headings': retrieved_docs[idx]['headings'],'text':retrieved_docs[idx]['text']}) for idx in range(len(retrieved_docs))])
    context = "\n\n".join([retrieved_docs[idx]['text'] for idx in range(len(retrieved_docs))])
    print(f"{context}\n--------------------\n")
    search = generate()
    
    results = search.llm_query(query, context)
    print(results)
