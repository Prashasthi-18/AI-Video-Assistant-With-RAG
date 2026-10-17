from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.llm import get_llm
from core.vector_store import build_vector_store, get_retriever

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert meeting assistant. Answer the user's question
based ONLY on the transcript context provided below.

If the answer is not found in the context, say:
"I could not find this information in the transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from transcript:
{context}""",
    ),
    ("human", "{question}"),
])


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def build_rag_chain(transcript: str):

    vector_store = build_vector_store(transcript)

    retriever = get_retriever(vector_store, k=4)

    # Full LCEL RAG pipeline: question → retrieve → prompt → LLM → text
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | get_llm()
        | StrOutputParser()
    )

    return rag_chain


def ask_question(rag_chain, question: str) -> str:
    print(f"Question : {question}")
    answer = rag_chain.invoke(question)
    print(f"Answer : {answer}")
    return answer
