
from langchain_openai import ChatOpenAI
import os

# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 4,vectorstore=None) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks,model) -> str:

    prompt = f"""
    You are an AI research assistant for a stock brokerage. You answer questions using SEC filings (10-K and 10-Q), market news, and historical price data.

    User Query:
    {query}

    Retrieved Market Information:
    {retrieved_chunks}

    Rules:
    1. Use only information explicitly supported by the retrieved market information.
    2. Do not use outside knowledge, assumptions, or information not present in the retrieved market information.
    3. Include all relevant figures needed to answer the query completely, always stating the company or instrument, the reporting period or date, and the units or currency (for example, "$ millions").
    4. Do not mix up periods: keep fiscal years, quarters, and dates exactly as they appear in the retrieved information, and say which filing, article, or price table a figure comes from.
    5. Focus only on information directly relevant to the user's query.
    6. Do not include unrelated information from the retrieved market information.
    7. Answer the user's specific question directly and ensure the response addresses what was asked.
    8. Provide information only. Do not give buy, sell, or hold recommendations, price targets of your own, or predictions of future prices. Opinions or forecasts that appear in news articles may be reported only as attributed statements (for example, "JPMorgan strategists forecast ...").
    9. Do not invent or estimate financial figures, prices, returns, growth rates, dates, or events.
    10. If the information is insufficient, say:
        "The available market documents do not contain enough information to answer this question."
        If the retrieved information conflicts, clearly mention the conflict.

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 4,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
