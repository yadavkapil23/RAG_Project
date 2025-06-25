from vector_rag import query_vector_store
import wikipedia
from langchain_openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

wikipedia.set_lang("en")

async def get_smart_rag_response(query: str) -> str:
    # First try vector store (local documents)
    vector_answer = query_vector_store(query)
    if vector_answer:
        return f"[Local Document]\n{vector_answer}"
    
    # Next try Wikipedia
    try:
        summary = wikipedia.summary(query, sentences=5)
        prompt = f"""Use the following Wikipedia info to answer:\n\n{summary}\n\nQuestion: {query}\nAnswer:"""
        result = llm.generate([prompt])
        return f"[Wikipedia]\n{result.generations[0][0].text.strip()}"
    except wikipedia.exceptions.PageError:
        pass  # Move on to fallback
    except wikipedia.exceptions.DisambiguationError as e:
        return f"The query is ambiguous. Did you mean: {', '.join(e.options[:5])}?"

    # Fallback to pure LLM
    return f"[🤖 Fallback LLM]\n{llm.predict(f'Answer the question: {query}')}"
