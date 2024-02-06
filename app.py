import logging
import os

from flask import Flask, request, jsonify, send_file,make_response
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import PreProcessor
from haystack.utils import convert_files_to_docs
from haystack.nodes import EmbeddingRetriever
from haystack.pipelines.standard_pipelines import DocumentSearchPipeline
from haystack.nodes import PromptNode, PromptTemplate, AnswerParser
from haystack.pipelines import Pipeline

app = Flask(__name__)

# Load Haystack configuration (replace with your indexing and model paths)
doc_dir = "docs"
model_api_key = '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'
pdf_docs_dir = 'pdf_docs'

if not os.path.exists("document_store.faiss"):
    # If "document_store.faiss" doesn't exist, create a new document store
    all_docs = convert_files_to_docs(dir_path=doc_dir)
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=False,
        split_by="word",
        split_length=50,
        split_respect_sentence_boundary=True,
    )
    docs = preprocessor.process(all_docs)
    document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")
    document_store.write_documents(docs)
    retriever = EmbeddingRetriever(
        document_store=document_store,
        api_key=model_api_key,
        embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1"
    )
    document_store.update_embeddings(retriever)
    document_store.save('document_store.faiss')
else:
    # If "document_store.faiss" exists, load the existing document store
    document_store = FAISSDocumentStore.load("document_store.faiss")
    retriever = EmbeddingRetriever(
        document_store=document_store,
        api_key=model_api_key,
        embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1"
    )

rag_prompt = PromptTemplate(
    prompt="""Synthesize a comprehensive answer from the following text for the given question.
                             Provide a clear and concise response that summarizes the key points and information presented in the text.
                             \n\n Related text: {join(documents)} \n\n Question: {query} \n\n Answer:""",
    output_parser=AnswerParser(),
)
prompt_node = PromptNode(model_name_or_path="tiiuae/falcon-7b",api_key=model_api_key, default_prompt_template=rag_prompt)

pipe = Pipeline()
pipe.add_node(component=retriever, name="retriever", inputs=["Query"])
pipe.add_node(component=prompt_node, name="prompt_node", inputs=["retriever"])

@app.route('/download/<filename>', methods=['GET'])
def download_pdf(filename):
    file_path = os.path.join(pdf_docs_dir, filename)

    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            # Read the existing PDF file content
            pdf_content = file.read()
            print(pdf_content)

        # Return PDF as response
        response = make_response(pdf_content)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename={filename}'
        return response
    else:
        return jsonify({"error": "File not found"}), 404

@app.route("/ask", methods=["POST"])
def answer_question():

    query = request.json["query"]
    answer = pipe.run(query=query)
    return jsonify({"answer": answer['answers'][0].answer})

@app.route('/search_document', methods=['POST'])
def search_document():
    query = request.json['query']
    pipeline = DocumentSearchPipeline(retriever)
    result = pipeline.run(query, params={"Retriever": {"top_k": 5}})
    documents = [
        {
            'meta': {
                'name': document.meta['name'].replace('.txt', '.pdf')
            }
        }
        for document in result['documents']
    ]

    result['documents'] = documents
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)