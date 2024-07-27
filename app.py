import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from pypdf import PdfReader, PdfWriter
from tempfile import NamedTemporaryFile
import base64
from htmlTemplates import expander_css, css, bot_template, user_template
from openai import OpenAI
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

# load the API keys
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

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
        ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3),
        # returns top 2 most relevant documents 
        retriever=db.as_retriever(search_kwargs={"k":2}),
        # returns actual documents from which the response was derived for later reference 
        return_source_documents=True
    )

    # return conversational chain object
    return conversation_chain

# Method for Handling User Input
def handle_userinput(query, expander):
    # generate a response using conversational chain object 
    response = st.session_state.conversation({"question": query, "chat_history": st.session_state.chat_history}, return_only_outputs=True)
    
    # append query and recieved response to chat history to session
    st.session_state.chat_history += [(query, response['answer'])]

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

    col1, col2 = st.columns([1,1])

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
        
        st.subheader("Summary")


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
                    with open(temp2.name, "rb") as f:
                        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

                        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}#page={3}"\
                            width="100%" height="900" type="application/pdf"></iframe>'
                    
                        st.markdown(pdf_display, unsafe_allow_html=True)


    



if __name__ == '__main__':
    main()

