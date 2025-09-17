# app/services/rag_pipeline.py

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from llama_cpp import Llama
from langchain.schema.runnable import Runnable
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain

# -----------------------
# Wrapper for llama_cpp to satisfy LangChain Runnable
# -----------------------
class LlamaWrapper(Runnable):
    def __init__(self, model_path, n_ctx=2048, n_threads=4, temperature=0.0):
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            temperature=temperature,
            verbose=False
        )

    def invoke(self, input, config=None, **kwargs):
        """
        Convert LangChain's input to string, call llama-cpp,
        and return only the text for LangChain.
        """
        if hasattr(input, "to_string"):
            prompt = input.to_string()
        elif isinstance(input, dict) and "question" in input:
            prompt = str(input["question"])
        else:
            prompt = str(input)

        response = self.model(prompt)

        # Extract generated text
        if isinstance(response, dict) and "choices" in response:
            text = response["choices"][0]["text"]
        else:
            text = str(response)

        return text

# -----------------------
# Load ChromaDB embeddings
# -----------------------
persist_directory = "./chroma_db"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma_db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
retriever = chroma_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# -----------------------
# Load LLaMA locally
# -----------------------
llm = LlamaWrapper(
    model_path="models/llama-3.1.gguf",
    n_ctx=2048,
    n_threads=4,
    temperature=0.0
)

# -----------------------
# Add conversation memory
# -----------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)


# -----------------------
# Define prompt template
# -----------------------
template = """
You are a helpful and accurate **medical assistant chatbot**.
Answer the user’s medical questions clearly, in English only, and based strictly on the retrieved context.

If you don’t know the answer, say:
"I’m not sure based on the available information. Please consult a qualified doctor."

Use the chat history for continuity.

Chat History:
{chat_history}

Context from medical documents:
{context}

User Question:
{question}

Your Answer (in English):
"""

prompt = ChatPromptTemplate.from_template(template)


# -----------------------
# Build Conversational Retrieval QA chain with memory
# -----------------------
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=True,
    output_key="answer"
)

# -----------------------
# Function for Streamlit UI
# -----------------------
def ask_question(user_query: str):
    """
    Takes a string query from the user,
    runs the RAG QA chain, and returns
    the answer + source documents.
    """
    if not user_query.strip():
        return "Please enter a valid question.", []

    result = qa_chain.invoke({"question": user_query})
    answer = result.get("answer", "No answer found.")
    sources = result.get("source_documents", [])
    return answer, sources

# -----------------------
# Optional: Quick test
# -----------------------
if __name__ == "__main__":
    test_query1 = "What exercises are recommended for hypothyroidism patients to lose weight?"
    answer, sources = ask_question(test_query1)
    print("Answer:\n", answer)

    test_query2 = "Can these exercises be done by someone with knee pain?"
    answer, sources = ask_question(test_query2)
    print("\nAnswer with memory:\n", answer)
