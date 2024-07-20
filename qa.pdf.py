# %%
!pip install transformers sentence-transformers langchain langchain-community langchain-openai faiss-cpu unstructured unstructured[pdf]

# %%
!pip freeze > requirements.txt

# %%
!which python
!python --version

# %%
import os

from langchain_openai import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA


# %%
from dotenv import load_dotenv
import os

# Carica le variabili di ambiente dal file .env
load_dotenv()

# Ora puoi accedere alle variabili di ambiente
openai_api_key = os.getenv("OPENAI_API_KEY")

# %%
#llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)


# %%
# load the document
path_file="/Users/desi76/repo-git-nugh/doc-qa/pdfs/hoder.pdf"
loader = UnstructuredFileLoader(path_file)
documents = loader.load()

# %%
type(documents[0])

# %%
# create text chunks

text_splitter = CharacterTextSplitter(separator='/n',
                                      chunk_size=2000,
                                      chunk_overlap=200)

text_chunks = text_splitter.split_documents(documents)

# %%
# Inizializza HuggingFaceEmbeddings con il modello desiderato
model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Sostituisci con il modello Hugging Face desiderato
embeddings = HuggingFaceEmbeddings(model_name=model_name)
# vector embedding for text chunks
knowledge_base = FAISS.from_documents(text_chunks, embeddings)
# Stampa alcune informazioni sulla knowledge base
print("Knowledge Base creata con successo.")

# %%
# chain for aq retrieval
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=knowledge_base.as_retriever())

# %%

# List of questions
questions = [
    "Qual è il background dell'articolo",
    "Quali sono gli obiettivi",
    "Qual è la metodologia",
    "Quali sono gli strumenti di rilevamento dei dati",
    "Quali sono i risultati",
    "Qual è la discussione",
    "Quali sono le conclusioni"
]

# Initialize an empty list to store the results
results = []

# Iterate through the list of questions and get the responses
for question in questions:
    response = qa_chain.invoke({"query": question})
    results.append(response["result"])

# Define the output file path
output_file = "results.txt"

# Write the results to a text file
with open(output_file, "w") as file:
    for i, result in enumerate(results):
        file.write(f"Question {i + 1}:\n{questions[i]}\n")
        file.write(f"Response {i + 1}:\n{result}\n\n")

print(f"Results have been written to {output_file}")

# %%
print(results)


