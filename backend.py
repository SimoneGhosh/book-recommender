from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

#from langchain.chains import ConversationalRetrievalChain
from langchain_classic.chains import ConversationalRetrievalChain
#from langchain.memory import ConversationBufferMemory
from langchain_classic.memory import ConversationBufferMemory

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(
    #model="gpt-4o-mini",
    #temperature=0.7,
    #api_key=os.getenv("OPENAI_API_KEY")

    model="openai/gpt-4o-mini", # Note: OpenRouter often uses 'provider/model' format
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost:3000", # Required by OpenRouter
        "X-Title": "Book RAG App",
    }
)

# Load the vectorstore you created
#embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    model="openai/text-embedding-3-small" # Ensure this matches your saved index
)

vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

# Create a retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Set up memory for conversation history
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# Create the conversational chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True
)

def get_chatbot_response(user_question: str) -> str:
    """
    Get a response from the chatbot based on the user's question.
    """
    result = qa_chain({"question": user_question})
    return result["answer"]
