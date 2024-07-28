import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.rl_config import defaultPageSize
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Read all PDF files and return text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

# Get embeddings for each chunk
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Get conversational chain
def get_conversational_chain():
    prompt_template = """
    Allow to generate a contract based on the provided context. Answer the question as detailed as possible from the provided context. Allow comparisons between documents, provide summaries, and ensure all details are included.

    Context: {context}

    Question: {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", client=genai, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

# Clear chat history
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Upload some PDFs and ask me a question."}]

# Handle user input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    if "create contract" in user_question.lower():
        response = chain.invoke({"input_documents": docs, "question": "Extract contract details based on the provided context and help create a contract."}, return_only_outputs=True)
        details = response['output_text']  # Adjust based on actual response structure
        file_path = generate_legal_document(details)
        return f"Contract has been created. Download it [here]({file_path})"
    else:
        response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response['output_text']

# Generate contract PDF
def generate_legal_document(details, title="Employment Agreement", file_path="legal_document.pdf"):

    BottomMargin = 0.4 * inch
    TopMargin = 0.4 * inch
    LeftMargin = 0.4 * inch
    RightMargin = 0.4 * inch

    doc = SimpleDocTemplate(file_path, pagesize=A4,
                            rightMargin=RightMargin, leftMargin=LeftMargin,
                            topMargin=TopMargin, bottomMargin=BottomMargin)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        name='Title',
        fontSize=18,
        leading=22,  # Reduced leading for title
        alignment=1,  # Center alignment
        spaceAfter=12  # Reduced space after title
    )
    normal_style = ParagraphStyle(
        name='Normal',
        parent=styles['Normal'],
        fontSize=12,
        leading=18,  # Reduced leading for content
        alignment=4  # Justified alignment for content
    )
    bold_style = ParagraphStyle(
        name='Bold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold'
    )

    content = []

    # Add uploaded images
    uploaded_images_folder = "uploaded_images"
    if os.path.exists(uploaded_images_folder):
        for image_file in os.listdir(uploaded_images_folder):
            image_path = os.path.join(uploaded_images_folder, image_file)
            if os.path.isfile(image_path):
                image = Image(image_path, width=2*inch, height=1*inch)
                content.append(image)
                content.append(Spacer(1, 0.25 * inch))  # Reduced spacing after image

    # Add title
    content.append(Spacer(1, 0.25 * inch))  # Reduced space before title
    content.append(Paragraph(title, title_style))

    # Add content
    content.append(Spacer(1, 0.25 * inch))  # Reduced space after title
    for line in details.split('\n'):
        if "**" in line:
            parts = line.split("**")
            line = ''.join([f'<b>{part}</b>' if i % 2 == 1 else part for i, part in enumerate(parts)])
        content.append(Paragraph(line, normal_style))
        content.append(Spacer(1, 0.05 * inch))  # Reduced spacing between lines

    doc.build(content)
    return file_path



def main():
    st.set_page_config(page_title="Gemini PDF Chatbot", page_icon="🤖")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

        st.subheader("Upload Company Images")
        company_images = st.file_uploader("Upload company images (optional)", accept_multiple_files=True)
        if company_images:
            uploaded_images_folder = "uploaded_images"
            os.makedirs(uploaded_images_folder, exist_ok=True)
            for image in company_images:
                image_path = os.path.join(uploaded_images_folder, image.name)
                with open(image_path, "wb") as f:
                    f.write(image.getbuffer())
            st.success("Images uploaded successfully")

        contract_title = st.text_input("Enter the title for the legal document:", "Employment Agreement")
        legal_text = st.text_area("Enter text for legal document:")
        if st.button("Generate Legal Document"):
            pdf_file_path = generate_legal_document(legal_text, title=contract_title)
            with open(pdf_file_path, "rb") as f:
                st.download_button(
                    label="Download Legal Document",
                    data=f,
                    file_name="legal_document.pdf",
                    mime="application/pdf"
                )
            st.success("Legal document generated and ready for download")

    st.title("Chat with PDF files using Gemini🤖")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Upload some PDFs and ask me a question."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                placeholder = st.empty()
                full_response = ''.join(response['output_text']) if isinstance(response, dict) else response
                placeholder.markdown(full_response)
        if response is not None:
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
