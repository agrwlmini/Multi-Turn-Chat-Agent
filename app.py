import os
from dotenv import load_dotenv
from creds import OPENAI_API_KEY,AZURE_OPENAI_ENDPOINT, OPENAI_DEPLOYMENT_NAME, OPENAI_API_VERSION,OPENAI_EMBEDDING_ENDPOINT_NAME,OPENAI_EMBEDDING_MODEL_NAME
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain
from PyPDF2 import PdfReader
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from templates import user_template, bot_template

# Load environment variables from .env file
load_dotenv()

# Initialize the language model with OpenAI
llm = AzureChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=OPENAI_DEPLOYMENT_NAME,
    openai_api_version=OPENAI_API_VERSION,validate_base_url=False
)

#define the embeddings
embeddings = AzureOpenAIEmbeddings(openai_api_key=OPENAI_API_KEY,
                                       azure_endpoint=OPENAI_EMBEDDING_ENDPOINT_NAME,
                                       model=OPENAI_EMBEDDING_MODEL_NAME,
                                       openai_api_version=OPENAI_API_VERSION,
                                       chunk_size=1, validate_base_url=False)

#setup the memory
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# Function to handle user input and display chat messages
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{message}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{message}}", message.content), unsafe_allow_html=True)


# Main function to run the Streamlit app
def main():

    st.set_page_config(page_title="Multi Turn Chat Agent")
    st.header("Multi Turn Chat Agent")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    user_question = st.chat_input("Ask any question related to the documents")
    if user_question:
        handle_userinput(user_question)
    
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload PDFs here and click on 'Data processing'", type=("pdf"), accept_multiple_files=True) 
                    
        if st.button("Data Processing"):
            with st.spinner("Processing the data... This may take a while⏳"):
                text = ""
                for page in uploaded_file:
                    pdf_reader = PdfReader(page)
                    for p in pdf_reader.pages:
                        text += p.extract_text()

                text_splitter = CharacterTextSplitter(
                                separator="\n",
                                chunk_size=1000,
                                chunk_overlap=200,
                                length_function=len
                            )
                chunks = text_splitter.split_text(text)
                vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
                # create a conversation chain with the vectorstore
                conversation_chain = ConversationalRetrievalChain.from_llm(
                                        llm=llm,
                                        retriever=vectorstore.as_retriever(),
                                        memory=memory
                                    )
                st.session_state.conversation = conversation_chain 
                st.text("Data Processing Complete!!")        

if __name__ == '__main__':
    main()
