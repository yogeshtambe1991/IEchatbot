### Import Libraries
from typing import Any
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

### PDF Reading and extracting table and text information
raw_pdf_elements = partition_pdf("ast_sci_data_tables_sample.pdf",
    # get bounding boxes for tables
    infer_table_structure=True,
    # Post processing to aggregate text once we have the title
    chunking_strategy="by_title",
    # max chars in chunk
    max_characters=800,
    # new chunk after 760 chars
    new_after_n_chars=760,
    # keep chars more than 400                             
    combine_text_under_n_chars=400,
)

### Create a dictionary to store counts of each type
category_counts = {}
for element in raw_pdf_elements:
    category = str(type(element))
    if category in category_counts:
        category_counts[category] += 1
    else:
        category_counts[category] = 1
unique_categories = set(category_counts.keys())
print(category_counts)

class Element(BaseModel):
    type: str
    text: Any
    
### Categorize by type
categorized_elements = []
for element in raw_pdf_elements:
    if "unstructured.documents.elements.Table" in str(type(element)):
        categorized_elements.append(Element(type="table", text=str(element)))
    elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
        categorized_elements.append(Element(type="text", text=str(element)))

### Tables element count
table_elements = [e for e in categorized_elements if e.type == "table"]
print(len(table_elements))

### Text element count
text_elements = [e for e in categorized_elements if e.type == "text"]
print(len(text_elements))

### Text and Table summarization
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Prompt for summarization
prompt_text = """You are an assistant tasked with summarizing tables and text. \
Give a concise summary of the table or text. Table or text chunk: {element} """
prompt = ChatPromptTemplate.from_template(prompt_text)

# Summary chain
model = ChatOllama(model="llama2:7b-chat")
summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

# Apply to text
texts = [i.text for i in text_elements if i.text != ""]
text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})

# Apply to tables
tables = [i.text for i in table_elements]
table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

import uuid
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Vectorstore db with embedding
vectorstore = Chroma(
    collection_name="summaries", embedding_function=GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf")
)

# The storage layer for the parent documents
store = InMemoryStore()
id_key = "doc_id"

# The retriever
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

# Add texts
doc_ids = [str(uuid.uuid4()) for _ in texts]
summary_texts = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(text_summaries)
]
retriever.vectorstore.add_documents(summary_texts)
retriever.docstore.mset(list(zip(doc_ids, texts)))

# Add tables
table_ids = [str(uuid.uuid4()) for _ in tables]
summary_tables = [
    Document(page_content=s, metadata={id_key: table_ids[i]})
    for i, s in enumerate(table_summaries)
]
retriever.vectorstore.add_documents(summary_tables)
retriever.docstore.mset(list(zip(table_ids, tables)))

# ### Multimodal RAG
# from langchain_core.runnables import RunnablePassthrough
# # Prompt template
# template = """Answer the question based only on the following context, which can include text and tables:
# {context}
# Question: {question}
# """
# prompt = ChatPromptTemplate.from_template(template)

# # LLM Model
# model = ChatOllama(model="llama2:7b-chat")

# # RAG pipeline
# chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt
#     | model
#     | StrOutputParser()
# )

print("Started Chatbot Implementation")
import streamlit as st
import random
import time

# st.title("IE Chatbot")

# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Accept user input
# if prompt := st.chat_input("What's up?"):
#     # Display user message in chat message container
#     with st.chat_message("user"):
#         st.markdown(prompt)
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     response = chain.invoke(prompt)
#     # Display assistant response in chat message container
#     with st.chat_message("assistant"):
#         st.markdown(response)
#     # Add assistant response to chat history    
#     st.session_state.messages.append({"role": "assistant", "content": response})

def get_response(question, context):
    # Prompt template
    template = """Answer the question based only on the following context, which can include text and tables:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # LLM Model
    model = ChatOllama(model="llama2:7b-chat")
    chain = prompt | model | StrOutputParser()
    return chain.stream({
        "context": context,
        "question": question,
    })

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = st.write_stream(get_response(user_query, st.session_state.chat_history))

    st.session_state.chat_history.append(AIMessage(content=response))