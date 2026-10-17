# Action items, key decisions, open questions

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel

from core.llm import get_llm


def build_chain(system_prompt: str):
    return (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | ChatPromptTemplate.from_messages([
            ("system", system_prompt + "\nReturn only the list (or the 'none found' sentence) — no extra commentary."),
            ("human", "{text}"),
        ]) | get_llm() | StrOutputParser()
    )


def extract_action_items(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the transcript (a meeting or video), "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )
    return chain.invoke(transcript)


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the transcript (a meeting or video), "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "From the transcript (a meeting or video), extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    return chain.invoke(transcript)


def extract_all(transcript: str) -> dict:
    """Run the three extractions in parallel."""
    parallel = RunnableParallel(
        action_items=RunnableLambda(extract_action_items),
        key_decisions=RunnableLambda(extract_key_decisions),
        open_questions=RunnableLambda(extract_questions),
    )
    return parallel.invoke(transcript)
