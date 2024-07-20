import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
import tempfile
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)


# Inizializzazione delle variabili di sessione di Streamlit
if 'initialized' not in st.session_state:
    st.session_state.update({
        'initialized': True,
        'results': [],
        'questions': []
    })

# Configurazione di Streamlit
st.title("Interfaccia di QA su documenti PDF")
st.write("Carica un file PDF, scegli il modello e invia le tue domande.")

# Seleziona il modello di embedding e il modello GPT
model_name = st.selectbox("Scegli un modello Hugging Face", ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L12-v2"])
gpt_model = st.selectbox("Scegli un modello LLM", ["llama3", "phi3","gemma2"])
temperature = st.slider("Regola la temperatura", 0.0, 1.0, 0.3)

# Carica il file PDF
uploaded_file = st.file_uploader("Carica un file PDF", type="pdf")

# Processo il file caricato
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name

    with st.spinner('Caricamento del documento...'):
        loader = UnstructuredFileLoader(temp_file_path)
        documents = loader.load()

    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=3000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    knowledge_base = FAISS.from_documents(text_chunks, embeddings)
    st.success("Knowledge base pronta.")

 
    llm = ChatOpenAI(base_url="http://localhost:11434/v1", temperature=temperature,  api_key="not-needed", model_name=gpt_model)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=knowledge_base.as_retriever())
 
    st.session_state.questions = st.text_area("Inserisci le domande (una per riga)").split('\n')

    if st.button("Ottieni risposte"):
        st.session_state.results = []
        with st.spinner('Elaborazione delle risposte...'):
            for question in st.session_state.questions:
                if question.strip():
                    response = qa_chain.invoke({"query": question.strip()})
                    st.session_state.results.append((question.strip(), response["result"]))
        for question, result in st.session_state.results:
            st.write(f"**Domanda:** {question}")
            st.write(f"**Risposta:** {result}")

# Gestione del salvataggio dei risultati su file e creazione di un pulsante per il download
filename = st.text_input("Nome del file per salvare le risposte:", "results.txt")
if st.button("Salva risposte su file"):
    if st.session_state.results:
        with open(filename, "w") as file:
            for i, (question, result) in enumerate(st.session_state.results):
                file.write(f"Domanda {i + 1}:\n{question}\n")
                file.write(f"Risposta {i + 1}:\n{result}\n\n")

        # Dopo aver salvato i risultati, mostra un pulsante di download
        with open(filename, "rb") as f:
            st.download_button(
                label="Scarica i risultati",
                data=f,
                file_name=filename,
                mime="text/plain"
            )
        st.success(f"Risultati salvati in {filename}")
    else:
        st.error("Nessun risultato da salvare.")