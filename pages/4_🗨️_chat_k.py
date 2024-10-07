# 해당 논문에 대해 이야기 할 수 있는 챗봇 만들기 
import os 
import glob
import streamlit as st
from langchain_core.messages.chat import ChatMessage
from utils.RagPipeline2 import Ragpipeline

from utils.config import config

# OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
# UPSTAGE_API_KEY = st.secrets["UPSTAGE_API_KEY"]
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# os.environ["UPSTAGE_API_KEY"] = UPSTAGE_API_KEY

import uuid
# 프롬프트 비용이 너무 많이 소요되는 것을 방지하기 위해
MAX_MESSAGES_BEFORE_DELETION = 6

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id

# 파일 업로드 전용 폴더: 임시로 저장 
root_dir = ".cache/files"

# ------------------------------------- 구성: 제목 -------------------------------------------------
st.title("챗봇에게 물어보기")
st.markdown("논문의 핵심 또는 핵심 용어가 무엇인지 질문해 보세요.")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    # 아무런 파일을 업로드 하지 않을 경우
    st.session_state["chain"] = None

# ------------------------------------- 구성: 사이드 바 -------------------------------------------------

pdfs = [None] + [os.path.basename(name).split('.')[0] for name in glob.glob(root_dir + '/*.pdf')]

with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    # 모델 선택 메뉴
    selected_model = st.selectbox(
        "LLM 선택", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"], index=0
    )
    
    # 모델 선택 메뉴
    selected_paper = st.selectbox(
        "준비된 논문 선택", pdfs, index=0
    )

# ----------------------------------------

# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    
    if len(st.session_state["messages"]) >= MAX_MESSAGES_BEFORE_DELETION:
        # Remove the first two messages
        del st.session_state["messages"][0]
        del st.session_state["messages"][0] 
    
    st.session_state["messages"].append(ChatMessage(role=role, content=message))



# 논문 골랐을 때 
if selected_paper:

    config["llm_predictor"]["model_name"] = selected_model
    db_path = os.path.join(root_dir, selected_paper + '/db').replace('\\','/')

    chain = Ragpipeline(db_path, config)
    
    st.session_state["chain"] = chain
    
    print('langchain 준비 완료')
    
    
# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []
    
# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("논문에 대해 궁금한 내용을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()


# 만약에 사용자 입력이 들어오면...
if user_input:
    # chain 을 생성
    chain = st.session_state["chain"]
    
    if chain is not None:
        # 사용자의 입력
        st.chat_message("user").write(user_input)
        # 스트리밍 호출
        response = chain.answer_generation(user_input, session_id)
        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()

            ai_answer = ""
            for chunk in response:
                ai_answer += chunk + " "
                container.markdown(ai_answer + "▌")
                container.markdown(ai_answer)

        # 대화기록을 저장한다.
        add_message("user", user_input)
        add_message("assistant", ai_answer)
        
        print('st.session_state["messages"]')
        print(st.session_state["messages"])
        
    else:
        # 파일을 업로드 하라는 경고 메시지 출력
        warning_msg.error("파일을 업로드 해주세요.")

