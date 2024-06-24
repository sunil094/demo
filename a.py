from langchain_huggingface import HuggingFaceEndpoint
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.chains import RetrievalQA

import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


sec_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
# llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7, token=sec_key)

st.markdown(
    """
    <style>
    .header {
        display: flex;
        align-items: center;
        padding: 20px;
    }
    .header img {
        max-width: 100px; 
        margin-top: 12px; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns((2.5,10))

with col1:
    st.markdown(
        """
        <div class="header">
            <img src="https://upload.wikimedia.org/wikipedia/commons/0/09/Mastek_logo.png" alt="Logo">
        </div>
        """,
        unsafe_allow_html=True
    )
    # st.image("Mastek_logo.png", width=100) 
with col2:
    st.title("Demo Mastek Chatbot")

st.markdown("---")

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

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# def get_conversation_chain(vectorstore):
#     llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7, token=sec_key)
#     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )
#     return conversation_chain



with st.sidebar:
    st.subheader("your doc")
    pdf_docs = st.file_uploader("choose here", accept_multiple_files=True)
    if st.button("process"):
        with st.spinner("processing"):

            raw_text = get_pdf_text(pdf_docs)

            text_chunks = get_text_chunks(raw_text)
            st.write(text_chunks)

            # vectorstore = get_vectorstore(text_chunks)

            # st.session_state.conversation = get_conversation_chain(vectorstore)
        

# if "conversation" not in st.session_state:
#     st.session_state.conversation = None
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = None


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
prompt = st.chat_input("What is up?")
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    st.write(retriever)
    
    llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7, token=sec_key)
    def pretty_print_docs(docs):
        for i, doc in enumerate(docs):
            st.write(f"Document {i+1}:")
            st.write(doc.page_content)
            st.write(f"Metadata: {doc.metadata}")
            st.write("\n" + "-" * 100 + "\n")
    results = retriever.get_relevant_documents(prompt)
    pretty_print_docs(results)


    compressor = FlashrankRerank()
    compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
    )
    def pretty_print_docs(docs):
        for i, doc in enumerate(docs):
            st.write(f"Document {i+1}:")
            st.write(doc.page_content)
            st.write(f"Metadata: {doc.metadata}")
            st.write("\n" + "-" * 100 + "\n")
    results1 = compression_retriever.get_relevant_documents(prompt)
    pretty_print_docs(results1)
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=compression_retriever)
    response = chain.invoke(prompt)


    # llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7, token=sec_key)
    # chain = load_qa_chain(llm=llm, chain_type="stuff")
    # response = chain.run(input_documents=query, question=prompt)
    
    # response = st.session_state.conversation({'question': prompt})
    # st.session_state.chat_history = response['chat_history']
    # latest_answer = response['answer']

    with st.chat_message("assistant"):
        st.markdown(response['query'])
        st.markdown(response['result'])
    st.session_state.messages.append({"role": "assistant", "content": response})


























# from langchain_huggingface import HuggingFaceEndpoint
# import streamlit as st

# import os

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# sec_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
# llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7, token=sec_key)

# st.write("Demo Bot Online")
# llm_string = st.text_input("Enter some text!")
# button_clicked = st.button("Ask!")

# if button_clicked:
#     if llm_string:
#         result = llm.invoke(llm_string)

#         st.write("LLM Result: ")
#         st.info(result)
#     else:
#         st.warning("Enter some text before sending")
