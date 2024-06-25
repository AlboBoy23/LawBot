# from llama_index.llms.huggingface import HuggingFaceLLM 
import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_together import Together
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="LawGPT")
col1, col2, col3 = st.columns([1,4,1])
with col2:
    st.image("C:\Abdul Saboor\Programing\TUKL Internship\LAW Bot\img.png")

st.markdown(
    """
    <style>
div.stButton > button:first-child {
    background-color: #ffd0d0;
}
div.stButton > button:active {
    background-color: #ff6262;
}
   div[data-testid="stStatusWidget"] div button {
        display: none;
        }
    
    .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    button[title="View fullscreen"]{
    visibility: hidden;}
        </style>
""",
    unsafe_allow_html=True,
)

if "app_stopped" not in st.session_state:
    st.session_state["app_stopped"] = False
elif st.session_state["app_stopped"]:
    st.session_state["app_stopped"] = False

def stopRunning():
    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"]})
    st.session_state["app_stopped"] = True


if st.session_state["app_stopped"]:
    st.stop()

def reset_conversation():
  st.session_state.messages = []
  st.session_state.memory.clear()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history",return_messages=True) 

embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1",model_kwargs={"trust_remote_code":True,"revision":"289f532e14dbbbd5a04753fa58739e9ba766f3c7"})
db = FAISS.load_local("C:\Abdul Saboor\Programing\TUKL Internship\LAW Bot_2\my_512_100", embeddings,
                      allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity",search_kwargs={"k": 4})

# If a question is unrelated to Pakistan Penal Code than tell user that you don't know the answer.

#prompt_template = """<s>[INST]You are a legal chat bot specializing in Pakistan Penal Code queries, your primary objective is to provide accurate and concise information based on the user's questions. You will prioritize the user's query and will adhere strictly to the instructions provided, offering relevant context from the knowledge base while avoiding unnecessary details. If a question falls outside the given context, you will refrain from utilizing the chat history and instead rely on your own knowledge base to generate an appropriate response. Do not generate questions on your own. Response only to Pakistan Penal Code related questions.
# prompt_template = """System: You are an assistant for question-answering tasks.
# If you don't know the answer, just say that you don't know. 
# Do not generate questions or context on your own.
# keep the answer concise and accurate.[/INST]
# CONTEXT: {context}
# CHAT HISTORY: {chat_history}
# [INST]QUESTION: {question}[/INST]
# Answer: 
# """

prompt_template = (
    "<s>[INST] <<SYS>>Context information is below.\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "You are an assistant for question-answering tasks related to Pakistan Penal Code. If you don't know the answer, just say that you don't know. Do not generate questions on your own. Give summarized answer. Only answer for the questions related to Pakistan Penal Code.<</SYS>>\n"
    "Query: {question}[/INST]\n"
    "Answer: "
)


# tokenizer = AutoTokenizer.from_pretrained(
#     "meta-llama/Meta-Llama-3-8B-Instruct"
# )
# stopping_ids = [
#     tokenizer.eos_token_id,
#     tokenizer.convert_tokens_to_ids("<|eot_id|>"),
# ]
prompt = PromptTemplate(template=prompt_template,
                        input_variables=['context', 'question'])

# You can also use other LLMs options from https://python.langchain.com/docs/integrations/llms. Here I have used TogetherAI API
load_dotenv(
    dotenv_path="C:\Abdul Saboor\Programing\TUKL Internship\LAW Bot\.devcontainer\.env")
TOGETHER_AI_API = os.getenv("TOGETHER_AI")

# Snowflake/snowflake-arctic-instruct
llm = Together(
    model="Austism/chronos-hermes-13b",
    temperature=0.1,
    max_tokens=400,
    repetition_penalty=1.1,
    together_api_key=f"{TOGETHER_AI_API}"
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=st.session_state.memory,
    retriever=db_retriever,
    combine_docs_chain_kwargs={'prompt': prompt}
)

for message in st.session_state.messages:
    with st.chat_message(message.get("role")):
        st.write(message.get("content"))

input_prompt = st.chat_input("Say something")

if input_prompt:
    with st.chat_message("user"):
        st.write(input_prompt)

    st.session_state.messages.append({"role":"user","content":input_prompt})

    with st.chat_message("assistant"):
        with st.status("Thinking 💡...",expanded=True):
            result = qa.invoke(input=input_prompt)
            
            message_placeholder = st.empty()

            full_response = "⚠️ **_Note: Information provided may be inaccurate._** \n\n\n"
        
        st.button("Stop", on_click=stopRunning)
        for chunk in result["answer"]:
            full_response+=chunk
            time.sleep(0.02)
            
            message_placeholder.markdown(full_response+" ▌")
        st.button('Reset All Chat 🗑️', on_click=reset_conversation)
    

    st.session_state.messages.append({"role":"assistant","content":result["answer"]})