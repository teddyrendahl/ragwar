import pathlib

import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import VectorStore
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

PROMPT_TEMPLATE = """\
You are a tournament official for a wargaming competition. Your responses should be factual and to the point.
Use direct quotes from the rulebook as much as possible when making your rulings.

Here is the appropriate context from the rulebook.
<context>
{context}
</context>

Answer the following question. If you do not know the answer, respond with "I can not make a determination on that matter".
<question>
{question}
</question>
"""

CHROMA_PATH = pathlib.Path(__file__).parent / "chroma"


def get_model_response(vectorstore: VectorStore, client: OpenAI, query: str):
    """Look in our vector store and ask OpenAI for a ruling based on k closest excerpts"""
    # Search for similar documents
    results = vectorstore.similarity_search_with_relevance_scores(query, k=3)

    # Create a prompt
    prompt = PROMPT_TEMPLATE.format(
        question=query,
        context="\n\n".join([r[0].page_content for r in results]),
    )

    # Request a response from OpenAI
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for chunk in stream:
        yield chunk.choices[0].delta.content or ""


def main():
    # Load our secrets
    load_dotenv()

    # Load our database if we haven't already
    if "db" not in st.session_state:
        st.session_state.db = Chroma(
            persist_directory=CHROMA_PATH.absolute().as_posix(),
            embedding_function=OpenAIEmbeddings(),
        )

    # Create our client if we have not already
    if "client" not in st.session_state:
        st.session_state.client = OpenAI()

    # Page setup
    st.set_page_config("40k Core Rules Assistant")
    st.image("assets/40k.png")
    st.markdown(
        '<div style="text-align: center"> Your tech-priest companion for the 40k Core Rules. Ask a question below. </div>',
        unsafe_allow_html=True,
    )
    user_question = st.text_input("user_question", label_visibility="hidden")
    if user_question:
        with st.chat_message(
            name="tech_priest", avatar="assets/tech_priest.png"
        ):
            # Stream the response from the OpenAI
            st.write_stream(
                get_model_response(
                    st.session_state.db, st.session_state.client, user_question
                )
            )


if __name__ == "__main__":
    main()
