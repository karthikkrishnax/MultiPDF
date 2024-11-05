import langchain.chains
import langchain.chains.combine_documents
import langchain.chains.history_aware_retriever
import langchain.chains.retrieval
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings , OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from htmlTemplates import css, bot_template, user_template
from langchain.prompts import PromptTemplate

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# def get_conversation_chain(vectorstore):
#     llm = ChatOpenAI()
#     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
#     history_aware_retriever = create_history_aware_retriever(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),  # Make sure your vectorstore supports this method
#         prompt= "your prompt here"  # Specify a prompt to generate a search query
#     )
#     conversation_chain = create_retrieval_chain(retriever=history_aware_retriever, combine_docs_chain=None)  # Define combine_docs_chain as needed
#     return conversation_chain

# from langchain.prompts import PromptTemplate

# def get_conversation_chain(vectorstore):
#     llm = ChatOpenAI()
#     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
#     # Define a prompt template with appropriate input variables
#     prompt_template = """
#     Based on the following question, please search for relevant information.
#     Question: {input}
#     """
#     prompt = PromptTemplate(input_variables=["input"], template=prompt_template)

#     # Create a history-aware retriever with the prompt and vectorstore retriever
#     history_aware_retriever = create_history_aware_retriever(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         prompt=prompt
#     )

#     # Define a simple combine function for documents
#     def combine_documents(documents):
#         return "\n".join([doc.content for doc in documents])  # Assuming documents have a `content` attribute

#     # Create the conversation chain with the custom combine function
#     conversation_chain = create_retrieval_chain(
#         retriever=history_aware_retriever,
#         combine_docs_chain=combine_documents  # Pass the combining function here
#     )

#     return conversation_chain


# def get_conversation_chain(vectorstore):
#     llm = ChatOpenAI()
#     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
#     prompt_template = """
#     Based on the following question, please search for relevant information.
#     input: {input}
#     """
#     prompt = PromptTemplate(input_variables=["input"], template=prompt_template)

#     # Create a history-aware retriever with the prompt and vectorstore retriever
#     history_aware_retriever = create_history_aware_retriever(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),  # Ensure vectorstore has as_retriever method
#         prompt=prompt
#     )
#     # Create the conversation chain, specifying a method for combining documents if needed
#     conversation_chain = create_retrieval_chain(
#         retriever=history_aware_retriever,
#         combine_docs_chain=None  # Define or replace with a document combination chain if needed
#     )

#     return conversation_chain


# def handle_user_input(user_question):
#     # Assuming st.session_state.conversation is now a RunnableBinding
#     if st.session_state.conversation:
#         response = st.session_state.conversation({'question': user_question})  # Check if this works
#         # If not, try using .run() or the appropriate method based on the new API
#         # response = st.session_state.conversation.run({'question': user_question})

#         st.session_state.chat_history = response['chat_history']

#         for i, message in enumerate(st.session_state.chat_history):
#             if i % 2 == 0:
#                 st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
#             else:
#                 st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)




from langchain.prompts import PromptTemplate
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()  # Initialize the language model
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    # Define a valid PromptTemplate with the correct input variable
    prompt_template = PromptTemplate(
        input_variables=["input"],
        template="Based on the following question, please search for relevant information: {input}"
    )

    # Create a history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=vectorstore.as_retriever(),  # Ensure that the vectorstore can return a retriever
        prompt=prompt_template  # Use the prompt_template instead of a string
    )

    # Create the conversation chain
    conversation_chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=None  # Replace with actual combining chain if needed
    )

    return conversation_chain

def handle_user_input(user_question):
    if st.session_state.conversation:  # Check if the conversation chain is set
        response = st.session_state.conversation({"input": user_question})  # Use "input" as the key
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)







def main():
    load_dotenv(dotenv_path=".env")
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")  # This should be the first Streamlit command

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your document:")
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and press on Process", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get PDF text
                raw_text = get_pdf_text(pdf_docs)
                
                # Get text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)
                
                # Create vector store
                vectorstore = get_vector_store(text_chunks)

                # Get conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == "__main__":
    main()
