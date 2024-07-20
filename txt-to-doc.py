import os
import docx
from docx.shared import Inches

def txt_to_doc(input_file, output_file):
    # Crea un nuovo documento Word
    doc = docx.Document()

    # Apre e legge il file di testo
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # Aggiunge il contenuto al documento Word
    doc.add_paragraph(content)

    # Salva il documento Word
    doc.save(output_file)

def convert_folder(input_folder, output_folder):
    # Crea la cartella di output se non esiste
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Itera su tutti i file nella cartella di input
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename[:-4] + ".doc")
            
            txt_to_doc(input_path, output_path)
            print(f"Convertito: {filename} -> {filename[:-4]}.doc")

# Esempio di utilizzo
input_folder = "/Users/desi76/repo-git-nugh/doc-qa/pdfs/riassunti txt"
output_folder = "/Users/desi76/repo-git-nugh/doc-qa/pdfs/riassunti doc"
convert_folder(input_folder, output_folder)
print("Conversione completata!")