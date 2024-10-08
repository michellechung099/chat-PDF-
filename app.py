__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma 
from pypdf import PdfReader, PdfWriter
from tempfile import NamedTemporaryFile
from htmlTemplates import expander_css, css, bot_template, user_template
from openai import OpenAI
import os

# load the API keys
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

# Process the Input PDF
def process_file(doc):
    # create embeddings object with HuggingFace embedding function
    model_name = "thenlper/gte-small"
    model_kwargs = {"device": "cpu"}
    encode_kwargs= {"normalize_embeddings": False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, 
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # create a vector store for PDF document (load it into Chroma)
    db = Chroma.from_documents(doc, embeddings)

    # process query and retrieve relevant information from db 
    conversation_chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, openai_api_key=openai_api_key),
        # returns top 2 most relevant documents 
        retriever=db.as_retriever(search_kwargs={"k":2}),
        # returns actual documents from which the response was derived for later reference 
        return_source_documents=True
    )

    return conversation_chain

def extract_relevant_quotes(docs, query):
    MODEL = "gpt-3.5-turbo"
    relevant_text = "\n".join([doc.page_content for doc in docs])
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": (
                "Your task is to help answer a question given in a document."
                "The first step is to extract quotes relevant to the question from the document, delimited by ####."
                "Please output the list of quotes using <quotes></quotes>."
                "Respond with No relevant quotes found! if no relevant quotes were found."
            )},
            {"role": "user", "content": f"####\n{relevant_text}\n####\nQuestion: {query}"},
        ],
        temperature=0, 
    )

    return response.choices[0].message.content

def generate_final_response(initial_response, relevant_quotes, query):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": (
                "Below is an initial response to a question based on a document. "
                "Please refine this response using the relevant quotes provided. "
                "Ensure that the answer is accurate, detailed, and helpful."
            )},
            {"role": "user", "content": f"####\nInitial Response: {initial_response}\n####"},
            {"role": "user", "content": f"Relevant Quotes: {relevant_quotes}\n####"},
            {"role": "user", "content": f"Question: {query}"}
        ],
        temperature=0,
    )

    return response.choices[0].message.content



# Method for Handling User Input
def handle_userinput(query, expander):
    # generate a response using conversational chain object 
    response = st.session_state.conversation({"question": query, "chat_history": st.session_state.chat_history}, return_only_outputs=True)
    initial_response = response["answer"]

    relevant_docs = response['source_documents']

    # Apply prompt chaining to refine the response
    if initial_response:
        extracted_quotes = extract_relevant_quotes(relevant_docs, query)
        final_response = generate_final_response(initial_response, extracted_quotes, query)
    else:
        final_response = "No relevant information found in the document."
    
    # append query and recieved response to chat history to session
    st.session_state.chat_history += [(query, final_response)]

    # retrieve referenced page number in response 
    st.session_state.N = list(response['source_documents'][0])[1][1]['page']

    # update chat in expander with chat history in session state
    for i, message in enumerate(st.session_state.chat_history): 
        expander.write(user_template.replace("{{MSG}}", message[0]), unsafe_allow_html=True)
        expander.write(bot_template.replace("{{MSG}}", message[1]), unsafe_allow_html=True)




def main(): 
    # configure Web-page Layout
    st.set_page_config(
        page_title="Interactive PDF Reader",
        page_icon=":books:",
        layout="wide",
    )

    # render css directly 
    st.write(css, unsafe_allow_html=True)

    # initialize conversation variable that will use prompt for querying the LLM 
    if "conversation" not in st.session_state:
        st.session_state["conversation"] = None 
    
    # history of current session's conversation
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # PDF page number that is referenced when answering the question 
    if "N" not in st.session_state: 
        st.session_state["N"] = 0

    col1, col2 = st.columns([1,1], gap="large")

    # every widget with a key is automatically added to Session State 
    with col1: 
        st.header("chat PDF :books:")

        user_question = st.chat_input(placeholder="Ask a question on the contents of the PDF after processing the document", key="question")

        expander1 = st.expander("Your Chat", expanded=True)
        with expander1:
            st.markdown(expander_css, unsafe_allow_html=True)

        st.subheader("Your Documents")
        pdf_doc = st.file_uploader("Upload a PDF and click Process", key="pdf_doc")

        if col1.button("Process", key="process"):
            with st.spinner("Processing"):
                if pdf_doc is not None:
                    # reset conversation and chat history and N
                    st.session_state["conversation"] = None 
                    st.session_state["chat_history"] = []
                    st.session_state["N"] = 0
                    # creates a temp file used for processing uploaded pdf for automatic cleanup
                    with NamedTemporaryFile(suffix="pdf") as temp:
                        bytes_data = pdf_doc.getvalue()
                        temp.write(bytes_data)
                        # ensure file pointer is at the beginning 
                        temp.seek(0)
                        # read and process contents of PDF file
                        loader = PyPDFLoader(temp.name)
                        pdf = loader.load()
                        # save the returned conversation chain prompt based on uploaded pdf
                        st.session_state.conversation = process_file(pdf)
                        st.markdown("Done processing")


    with col2:
        if user_question:
            handle_userinput(user_question, expander1)
            with NamedTemporaryFile(suffix="pdf") as temp:
                if pdf_doc is not None:
                    temp.write(pdf_doc.getvalue())
                    temp.seek(0)
                    reader = PdfReader(temp.name)
                
                pdf_writer = PdfWriter()
                # start and ending page numbers of PDF to be extracted 
                start = max(st.session_state.N-2, 0)
                end = min(st.session_state.N+2, len(reader.pages)-1) 
                while start <= end:
                    pdf_writer.add_page(reader.pages[start])
                    start+=1

                with NamedTemporaryFile(suffix="pdf") as temp2:
                    pdf_writer.write(temp2.name)
                    temp2.seek(0)
                    pdf_data = temp2.read() 
                    pdf_viewer(input=pdf_data, width=800, height=900)

if __name__ == '__main__':
    main()

