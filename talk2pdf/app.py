from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from htmlTemplates import css, bot_template, user_template
import streamlit as st 
from PyPDF2 import PdfReader
from dotenv import load_dotenv



def retrieve_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=0,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    LLM = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=LLM,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
    return conversation_chain


def user_input_handler(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
      if i % 2 == 0: #for every pair of message by user and AI
        st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
      else:
        st.write(bot_template.replace(
            "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
  
    load_dotenv()

    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    #check if conversation and chat history are in session state
    if "conversation" not in st.session_state:
      st.session_state.conversation = None
    if "chat_history" not in st.session_state:
      st.session_state.chat_history = None

    st.header("Chat with multiple PDFs 📚📟")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
      user_input_handler(user_question)

    with st.sidebar:
      st.subheader("Your documents")
      pdf_docs = st.file_uploader(
        "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
      if st.button("Process"):
        with st.spinner("Processing"):
          # get pdf text
          raw_text = retrieve_pdf_text(pdf_docs)

          # get the text chunks
          text_chunks = get_text_chunks(raw_text)

          # create vector store
          vectorstore = get_vectorstore(text_chunks)

          # create conversation chain
          st.session_state.conversation = get_conversation_chain(
              vectorstore)
 

main()