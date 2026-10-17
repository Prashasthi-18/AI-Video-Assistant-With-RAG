from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.llm import get_llm


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )

    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    """Map-reduce: summarise each chunk, then combine the partial summaries."""
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this portion of a video/meeting transcript concisely."),
        ("human", "{text}"),
    ])

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)
    print(f"Summarising {len(chunks)} chunk(s)...")

    # batch() sends the chunk requests in parallel instead of one by one
    chunk_summaries = map_chain.batch([{"text": chunk} for chunk in chunks])

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert meeting summarizer. Combine these partial summaries "
            "into one final professional summary in bullet points. "
            "Return only the summary in Markdown — no preamble, no title, no closing remarks.",
        ),
        ("human", "{text}"),
    ])

    combined_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | combined_prompt | llm | StrOutputParser()
    )

    return combined_chain.invoke(combined)


def generate_title(transcript: str) -> str:
    llm = get_llm()

    title_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) |
        ChatPromptTemplate.from_messages([
            (
                "system",
                "Based on the transcript, generate a short professional title "
                "(max 8 words). Only return the title as plain text — no quotes, no Markdown.",
            ),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )

    title = title_chain.invoke(transcript[:2000])
    return title.replace("*", "").replace("#", "").strip().strip("\"'").strip()
