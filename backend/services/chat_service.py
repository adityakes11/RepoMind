from src.database import get_thread, save_message
from langsmith import traceable
from src.llm.generator import AnswerGenerator
from src.pipeline import RAGPipeline
from src.rag.retriever import retrieve_documents


def _get_collection(thread_id, user_id):
    thread = get_thread(thread_id, user_id)
    if thread is None:
        raise ValueError("Thread not found")
    if not thread["collection_name"]:
        raise ValueError("This thread has no indexed repository")
    return thread["collection_name"]


@traceable(run_type="chain", name="ChatAnswer")
def answer_question(thread_id, question, user_id):
    collection_name = _get_collection(thread_id, user_id)
    save_message(thread_id, user_id, "user", question)
    result = RAGPipeline().run(
        question,
        collection_name,
        langsmith_extra={"metadata": {"thread_id": str(thread_id)}},
    )
    save_message(thread_id, user_id, "assistant", result["answer"])
    return result


@traceable(run_type="chain", name="ChatStream")
def stream_question(thread_id, question, user_id):
    collection_name = _get_collection(thread_id, user_id)
    save_message(thread_id, user_id, "user", question)
    documents = retrieve_documents(question, collection_name, 3)
    generator = AnswerGenerator()
    answer_parts = []
    try:
        for chunk in generator.generate_stream(
            question,
            documents,
            langsmith_extra={"metadata": {"thread_id": str(thread_id)}},
        ):
            answer_parts.append(chunk)
            yield chunk
    finally:
        if answer_parts:
            save_message(thread_id, user_id, "assistant", "".join(answer_parts))