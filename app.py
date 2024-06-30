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

# Carica le variabili di ambiente dal file .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Configura Streamlit
st.title("Document QA Interface")
st.write("Carica un file PDF, scegli il modello e fai delle domande.")

# Carica il file PDF
uploaded_file = st.file_uploader("Scegli un file PDF", type="pdf")

# Seleziona il modello di embedding
model_name = st.selectbox("Scegli un modello Hugging Face", 
                          ["sentence-transformers/all-MiniLM-L6-v2", 
                           "sentence-transformers/all-MiniLM-L12-v2"])

# Seleziona il modello GPT
gpt_model = st.selectbox("Scegli un modello GPT", 
                         ["gpt-3.5-turbo", "gpt-4"])

# Seleziona la temperatura
temperature = st.slider("Seleziona la temperatura", 0.0, 1.0, 0.3)

if uploaded_file is not None:
    # Salva il file caricato temporaneamente su disco
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name

    # Carica il documento
    with st.spinner('Caricamento del documento...'):
        loader = UnstructuredFileLoader(temp_file_path)
        documents = loader.load()

    # Crea text chunks
    with st.spinner('Creazione dei text chunks...'):
        text_splitter = CharacterTextSplitter(separator='\n',
                                              chunk_size=2000,
                                              chunk_overlap=200)
        text_chunks = text_splitter.split_documents(documents)

    # Inizializza HuggingFaceEmbeddings
    with st.spinner('Inizializzazione degli embeddings...'):
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        # Crea vector embedding per i text chunks
        knowledge_base = FAISS.from_documents(text_chunks, embeddings)

    st.success("Knowledge Base creata con successo.")

    # Inizializza ChatOpenAI con il modello selezionato e la temperatura
    llm = ChatOpenAI(model=gpt_model, temperature=temperature)

    # Crea chain per AQ retrieval
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=knowledge_base.as_retriever()
    )

    # Area per inserire le domande
    st.write("Inserisci le domande:")
    questions = st.text_area("Domande (una per riga)").split('\n')

    if st.button("Ottieni risposte"):
        # Inizializza una lista vuota per memorizzare i risultati
        results = []

        # Itera attraverso la lista delle domande e ottieni le risposte
        with st.spinner('Elaborazione delle risposte...'):
            for question in questions:
                if question.strip():  # Assicurati che la domanda non sia vuota
                    response = qa_chain.invoke({"query": question.strip()})
                    results.append((question.strip(), response["result"]))

        # Mostra le risposte
        for question, result in results:
            st.write(f"**Domanda:** {question}")
            st.write(f"**Risposta:** {result}")

        # Funzione per salvare le risposte su un file di testo
        def save_responses_to_file(results, filename="results.txt"):
            with open(filename, "w") as file:
                for i, (question, result) in enumerate(results):
                    file.write(f"Domanda {i + 1}:\n{question}\n")
                    file.write(f"Risposta {i + 1}:\n{result}\n\n")

        # Bottone per salvare le risposte su un file di testo
        if st.button("Salva risposte su file"):
            save_responses_to_file(results)
            st.success("Risposte salvate su 'results.txt'.")