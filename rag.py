from dotenv import load_dotenv
import os
import wikipedia

from langchain_openai import OpenAI

# Load environment variables (for OpenAI key)
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI LLM
llm = OpenAI(openai_api_key=openai_api_key)

# Set Wikipedia language
wikipedia.set_lang("en")

# 🔍 Wikipedia + LLM Response
async def get_wikipedia_rag_response(query: str) -> str:
    try:
        # Fetch Wikipedia summary
        summary = wikipedia.summary(query, sentences=5)

        # Form a prompt with Wikipedia data
        prompt = f"""Use the following Wikipedia information to answer the question as clearly as possible.

Wikipedia Context:
{summary}

Question: {query}
Answer:"""

        # Get response from LLM
        result = llm.generate([prompt])
        answer = result.generations[0][0].text if hasattr(result, "generations") else str(result)
        return answer.strip()

    except wikipedia.exceptions.DisambiguationError as e:
        return f"The query is ambiguous. Did you mean: {', '.join(e.options[:5])}?"
    except wikipedia.exceptions.PageError:
        return "No relevant Wikipedia page found for your query."
    except Exception as e:
        return f"An error occurred: {str(e)}"
