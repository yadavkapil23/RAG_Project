from vector_rag import query_vector_store
import wikipedia
from langchain_openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
wikipedia.set_lang("en")

async def get_smart_rag_response(query: str) -> str:
    print(" Received Query:", query)

    # First: Try Wikipedia
    try:
        summary = wikipedia.summary(query, sentences=5)
        print("Wikipedia summary found.")
        prompt = f"""Use the following Wikipedia information to answer the question as clearly as possible.

Wikipedia Context:
{summary}

Question: {query}
Answer:"""
        result = llm.generate([prompt])
        return f"[Wikipedia]\n{result.generations[0][0].text.strip()}"
    except wikipedia.exceptions.PageError:
        print("Wikipedia page not found.")
    except wikipedia.exceptions.DisambiguationError as e:
        return f"The query is ambiguous. Did you mean: {', '.join(e.options[:5])}?"

    # Second: Fallback to LLM (no context)
    try:
        print("Fallback: LLM with no context")
        fallback_prompt = f"You are a knowledgeable assistant. Please answer the following question clearly:\n\n{query}"
        llm_answer = llm.predict(fallback_prompt)
        if llm_answer and "not sure" not in llm_answer.lower():
            return f"[LLM Fallback]\n{llm_answer.strip()}"
    except Exception as e:
        print("Error during LLM fallback:", e)

    #Finally: Fallback to Local Documents
    try:
        print("Fallback: Local vector search")
        vector_answer = query_vector_store(query)
        if vector_answer:
            return f"[Local Document]\n{vector_answer}"
    except Exception as e:
        print("Error during local vector search:", e)

    return "Sorry, I couldn’t find any information to answer your question."
