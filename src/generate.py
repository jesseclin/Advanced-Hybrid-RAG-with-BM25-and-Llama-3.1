from ollama import Client
from rag_evaluator import evaluate_rag_relevance  # Assuming previous code is saved as evaluate_rag.py
#from rerankers import Reranker
import csv
import re
import sys
import json 

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

def read_csv_and_contate(filename: str) -> list[str]:
    """
    Reads the data from a CSV file named 'filename', concatenates items in each line into a single string separated by whitespace, adds this string to a list, and returns the list.
    
    Args:
        filename (str): The name of the CSV file to read.
        
    Returns:
        list[str]: A list where each element is a concatenated string from a line in the CSV file.
    """
    result = []
    
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            #if len(row)>0:
            #    print(row[0])
            concatenated = ' '.join(row)
            if re.search(r'[0-9A-Za-z]+', concatenated): 
                result.append([row[0], concatenated])
                print(concatenated)
            #sys.exit()    
    return result

def contains_string(paragraph, target_string, case_sensitive=False, whole_word=True):
    """
    Checks if a paragraph contains a specified string.

    Args:
        paragraph: The paragraph text (string).
        target_string: The string to search for.
        case_sensitive: Whether the search should be case-sensitive (default: True).
        whole_word: Whether to match only whole words (default: False).

    Returns:
        True if the string is found, False otherwise.
        Returns None if input types are invalid.
    """

    if not isinstance(paragraph, str) or not isinstance(target_string, str):
      return None

    if not case_sensitive:
        paragraph = paragraph.lower()
        target_string = target_string.lower()

    if whole_word:
        # Use regular expression for whole word matching
        pattern = re.escape(target_string) + r"\b"  # \b matches word boundaries
        match = re.search(pattern, paragraph)
        return bool(match) # convert match object to boolean
    else:
        return target_string in paragraph
        
if __name__ == '__main__':
    from retriever import retriver
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-c', '--collection', help='Qdrant\'s collection name')
    parser.add_argument('-s', '--csv', help='Csv for query')
    parser.add_argument('-j', '--json', help='Preparaed JSON database')  
    args = parser.parse_args() 
    collection_name = args.collection if args.collection is not None else "collection_bm25"
    print(collection_name)
    csv_name = args.csv if args.csv is not None else "test.csv"
    query_list = read_csv_and_contate(csv_name)
    dict_file = args.json if args.json is not None else "test-output.json"
    with open(dict_file, encoding='utf-8', mode='r') as file:
        data_dict = json.load(file)

    search = retriver(collection_name=collection_name)

    fail_cnt = 0
    total_cnt = 0
    fail_query = []
    for index, (prefix, query) in enumerate(query_list):

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
        #query = "FIGURE 1.24 Inefficient discrete gate implementation of AOI22 with transistor counts indicated"
        #query = "FIGURE 1.11 Inverter schematic (a) and symbol (b) Y = A"
        if prefix in data_dict.keys():
            retrievals_filtered = data_dict[prefix]
            scores = [99]*len(retrievals_filtered)
            #print(prefix)
        else:
            retrieved_docs = search.hybrid_search(query)
            retrievals = [retrieved_docs[idx]['text'] for idx in range(len(retrieved_docs))]
            retrievals_filtered = []
            for retrieval in retrievals:
                if contains_string(retrieval, prefix):
                    retrievals_filtered.append(retrieval)
        #retrievals_filtered = retrievals
            scores = evaluate_rag_relevance(query, retrievals_filtered)
        #context = "\n\n".join([str({'headings': retrieved_docs[idx]['headings'],'text':retrieved_docs[idx]['text']}) for idx in range(len(retrieved_docs))])
        #context = "\n\n".join([retrieved_docs[idx]['text'] for idx in range(len(retrieved_docs))])
        #print(scores)
        #print(f"{context}\n--------------------\n")
        #ranker = Reranker('flashrank')
        #results = ranker.rank(query=query, docs=retrievals)
        #print(results.top_k(1)[0].text)

        score_max = 0
        context = ""
        count = 0
        for idx, content in enumerate(retrievals_filtered):
            #print(f"Score: {scores[idx]}")
            #print("====================\n")
            #print(f"Content: {content}")
            #print("--------------------")
            #print("--------------------\n")
            if score_max < scores[idx]:
                context = content
                count = 1
                score_max = scores[idx]
            else:
                context += f" {content}"
                count += 1
    
        if score_max == 0:
            fail_cnt += 1
            fail_query.append(query)
            print(f"[{score_max}] [{prefix}]<-> [{context}]")

        #if score_max<80:
        #if not contains_string(context, prefix):
        #    fail_cnt += 1
        #    fail_query.append(query)
        #    print(f"[{score_max}] [{prefix}]<-> [{context}]")
            
        total_cnt += 1
        #search = generate()
        #print(count)
        #context = results.top_k(1)[0].text
        #results = search.llm_query(query, context)
        #print(results)

    for item in fail_query:
        print(item)
    print(f"{fail_cnt}/{total_cnt}")
