# from langchain_community.chat_models import ChatCohere
from langchain_cohere import ChatCohere
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
import json
import PyPDF2
import streamlit as st
import os
from dotenv import load_dotenv
import base64
import pandas as pd
from io import StringIO
import re
# os.environ['JAVA_HOME'] = './jdk'
st.set_page_config("Chatbot | ST","🤖")

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
COHERE_API_KEY = os.getenv('COHERE_API_KEY')






#Retrieve Schemes from CSV
def schemeRetrieve(path):
    df=pd.read_csv(path)
    df.columns=df.iloc[0].values
    df=df.drop(df.index[0])
    mf_scheme_list=["All Schemes"]
    for i in df['Scheme Name'].values:
        if "Schemes" in str(i):
            if str(i) not in mf_scheme_list:
                mf_scheme_list.append(str(i)) 
    return mf_scheme_list





# Using Cohere's embed-english-v3.0 embedding model
embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY, model="embed-english-v3.0")


# For OpenAI's gpt-3.5-turbo llm
# llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

# For Cohere's command-r llm
llm = ChatCohere(temperature=0, cohere_api_key=COHERE_API_KEY, model="command-r")


# For reading PDFs and returning text string
def read_pdf(files):
    file_content=""
    for file in files:
        # Create a PDF file reader object
        pdf_reader = PyPDF2.PdfReader(file)
        # Get the total number of pages in the PDF
        num_pages = len(pdf_reader.pages)
        # Iterate through each page and extract text
        for page_num in range(num_pages):
            # Get the page object
            page = pdf_reader.pages[page_num]
            file_content += page.extract_text()
    return file_content


#Download CSV
def download_df(content, filename='data.csv'):
    df = pd.read_csv(StringIO(content), sep="|", skipinitialspace=True)

    # Remove leading and trailing whitespaces from column names
    df.columns = df.columns.str.strip()
    df.drop(df.columns[df.columns.str.contains('Unnamed', case=False)], axis=1, inplace=True)
    # csv_bytes = content.encode()  # Convert string to bytes
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()    # Encode bytes to base64
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href


#-----------------------------------------------------------#
#------------------------💬 CHATBOT -----------------------#
#----------------------------------------------------------#
def chatbot():
    st.subheader("Ask questions from the PDFs")
    st.markdown("<br>", unsafe_allow_html=True)
    # Check if it is empty
    if st.session_state.book_docsearch:   
        prompt = st.chat_input("Say something")
        
        # Write previous converstions
        for i in st.session_state.conversation:
            user_msg = st.chat_message("human", avatar="💀")
            user_msg.write(i[0])
            computer_msg = st.chat_message("ai", avatar="✨")
            computer_msg.write(i[1])
            
        if prompt:
            exprompt=prompt                       #to store the previous prompt
            exprompt="For the "+", ".join(st.session_state.selected_scheme)+", "+exprompt
            exprompt+=" . Give the result in tabular format"          #want to show data in tabular form
            # promptcsv=prompt+" in csv format"   #for downloading csv version                                    
            user_text = f'''{prompt}'''
            user_msg = st.chat_message("human", avatar="💀")
            user_msg.write(user_text)

            with st.spinner("Getting Answer..."):
                # No of chunks the search should retrieve from the db
                chunks_to_retrieve = 5
                retriever = st.session_state.book_docsearch.as_retriever(search_type="similarity", search_kwargs={"k":chunks_to_retrieve})

                ## RetrievalQA Chain ##
                qa = RetrievalQA.from_llm(llm=llm, retriever=retriever, verbose=True)
                answer = qa({"query": exprompt})["result"]
                # answercsv = qa({"query": promptcsv})["result"]    #for generating csv file
                
                computer_text = f'''{answer}'''
                # print(answer)
                computer_msg = st.chat_message("ai", avatar="✨") 
                computer_msg.write(computer_text)
                
                #Download Data
                st.markdown(download_df(answer), unsafe_allow_html=True)
                
                # Showing chunks with score
                doc_score = st.session_state.book_docsearch.similarity_search_with_score(prompt, k=chunks_to_retrieve)
                with st.popover("See chunks..."):
                    st.write(doc_score)
                # Adding current conversation to the list.
                st.session_state.conversation.append((prompt, answer))   
    else:
        st.warning("Please upload a file")


            
# For initialization of session variables
def initial(flag=False):
    path="db"
    if 'existing_indices' not in st.session_state or flag:
        st.session_state.existing_indices = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    if ('selected_option' not in st.session_state) or flag:
        try:
            st.session_state.selected_option = st.session_state.existing_indices[0]
        except:
            st.session_state.selected_option = None
    
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'book_docsearch' not in st.session_state:
        st.session_state.book_docsearch = None
        
        
    if 'mf_schemes' not in st.session_state or flag:
        try:
            st.session_state.mf_schemes=schemeRetrieve(f"./{path}/table.csv")
        except:
            st.session_state.mf_schemes=None
            
    if ('selected_scheme' not in st.session_state) or flag:
        try:
            st.session_state.selected_scheme = st.session_state.mf_schemes[0]
        except:
            st.session_state.selected_scheme = None
            

def main():
    initial(True)
    # Streamlit UI
    st.title("💰 Mutual Fund Chatbot")
    
    # For showing the index selector
    file_list=[]
    for index in st.session_state.existing_indices:
        with open(f"db/{index}/desc.json", "r") as openfile:
            description = json.load(openfile)
            file_list.append(",".join(description["file_names"]))
    
    

    with st.popover("Select Index", help="👉 Select the datastore from which data will be retrieved"):
        st.session_state.selected_option = st.radio("Select a Document...", st.session_state.existing_indices, captions=file_list, index=0)

    st.write(f"*Selected Index* : **:orange[{st.session_state.selected_option}]**")
    
    if 'existing_indices' in st.session_state:
        
        st.session_state.mf_schemes=schemeRetrieve(f"./db/{st.session_state.selected_option}/table.csv")
        
        with st.popover("Select Scheme", help="👉 Select the Mutual Fund Scheme"):
        
            radio_scheme = st.multiselect("Select a Scheme...", st.session_state.mf_schemes)
            # if radio_scheme == "Other (Add Manually)...":
            #     radio_scheme = st.text_input("Write the name of the Schemes with comma")
            st.session_state.selected_scheme = radio_scheme
            # st.write(f"You have selected: {st.session_state.selected_scheme}")
        temp_scheme=", ".join(st.session_state.selected_scheme)
        st.write(f"*Selected Scheme* : **:green[{temp_scheme}]**")
        
    
    # Load the selected index from local storage
    if st.session_state.selected_option:
        st.session_state.book_docsearch = FAISS.load_local(f"db/{st.session_state.selected_option}", embeddings, allow_dangerous_deserialization=True)
        # Call the chatbot function
        chatbot()
    else:
        st.warning("⚠️ No index present. Please add a new index.")
        st.page_link("pages/Upload_Files.py", label="Upload Files", icon="⬆️")
            
            
 

            
main()