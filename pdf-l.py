import tempfile
import os
import streamlit as st

st.title('Ciao a tutti!!!')

st.write('This is a simple Streamlit app.')

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
question = st.text_input('Enter your question here:')

if uploaded_file is not None and len(question) > 0:
    st.write('You selected the following PDF file:', uploaded_file)
    st.write('You entered the following question:', question)
    temp_dir = tempfile.TemporaryDirectory()
    filename = os.path.join(temp_dir.name, uploaded_file.name)
    with open(filename, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    from langchain_community.document_loaders.pdf import PyPDFLoader
    loader = PyPDFLoader(filename)
    data = loader.load()

    from langchain_community.embeddings import GPT4AllEmbeddings
    model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
    embeddings = GPT4AllEmbeddings(model_name=model_name)
    
    from langchain_community.vectorstores import Chroma
    vectorstore = Chroma.from_documents(documents=data, embedding=embeddings, persist_directory="./chroma_db")

    if st.button("Ask!", type="primary"):
        from langchain_community.chat_models import ChatOllama
        from langchain import hub
        from langchain.chains import RetrievalQA

        llm = ChatOllama(model="llama3")
        prompt = hub.pull("rlm/rag-prompt")
        
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": prompt}
        )
        
        answer = qa_chain({"query": question})
        st.write(answer["result"])
