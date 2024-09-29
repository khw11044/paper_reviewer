

import os 
import glob
import streamlit as st
import base64
from utils.Classes import GraphState, LayoutAnalyzer
from utils.funcs import *
from utils.extracts import *
from utils.crops import *
from utils.creates import *
from utils.save import save_results

from utils.creates import create_text_trans_summary
from utils.vectordb import build_db

from dotenv import load_dotenv
load_dotenv()


OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
UPSTAGE_API_KEY = st.secrets["UPSTAGE_API_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["UPSTAGE_API_KEY"] = UPSTAGE_API_KEY

# analyzer = LayoutAnalyzer(os.environ.get("UPSTAGE_API_KEY"))
analyzer = LayoutAnalyzer(UPSTAGE_API_KEY)

import re
def st_markdown(markdown_string):
    parts = re.split(r"!\[(.*?)\]\((.*?)\)", markdown_string)
    for i, part in enumerate(parts):
        if i % 3 == 0:
            st.markdown(part)
        elif i % 3 == 1:
            title = part
        else:
            st.image(part)  # Add caption if you want -> , caption=title)

def display_pdf(file):
    # Opening file from file path

    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

# 파일 업로드 전용 폴더: 임시로 저장 
root_dir = ".cache/files"
os.makedirs(root_dir, exist_ok=True)

st.set_page_config(
    page_title="paper review AI Agent",
    page_icon=":star:"  # You can use emojis or provide a URL to a custom favicon image
)

# ------------------------------------- 구성: 제목 -------------------------------------------------
st.title("논문 원본 읽기")

# ------------------------------------- 구성: 사이드 바 -------------------------------------------------

pdfs = [None] + [os.path.basename(name).split('.')[0] for name in glob.glob(root_dir + '/*.pdf')]
# pdfs = [os.path.basename(name).split('.')[0] for name in glob.glob(root_dir + '/*.pdf')]

with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    # 파일 업로드
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])

    # 모델 선택 메뉴
    selected_model = st.selectbox(
        "LLM 선택", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"], index=0
    )
    
    # 모델 선택 메뉴
    selected_paper = st.selectbox(
        "준비된 논문 선택", pdfs, index=0
    )



# 파일이 업로드 되었을 때
# 파일을 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")     # 스피너 작업중인 것을 표기해줌
def main(file, selected_model):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f"./{root_dir}/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    # 1. 기본 문서 구조 분석
    state = GraphState(filepath=file_path, batch_size=10)
    state_out = split_pdf(state)
    state.update(state_out)

    # # 1.1 문서 구조 분석기를 통해 기본 분석 결과 저장 
    state_out = analyze_layout(analyzer, state)
    state.update(state_out)

    # 1.2 문서에 대한 메타데이터 추출 
    state_out = extract_page_metadata(state)
    state.update(state_out)

    # 1.3 문서 구조와 내용에 대한 html 내용 추출
    # 페이지별 정보를 추출 
    state_out = extract_page_elements(state)
    state.update(state_out)

    # 1.4 문서 요소 별 tag 추출
    state_out = extract_tag_elements_per_page(state)
    state.update(state_out)

    # 1.5 페이지 번호 추출 
    state_out = page_numbers(state)
    state.update(state_out)


    # 2.1 이미지를 추출하여 저장하고 위치를 저장 
    state_out = crop_image(state)
    state.update(state_out)

    # 2.2 표를 추출하여 저장하고 위치를 저장 
    state_out = crop_table(state)
    state.update(state_out)

    state_out = crop_equation(state)
    state.update(state_out)

    # 2.3 텍스트를 추출하고 저장하여 위치를 저장 
    state_out = extract_page_text(state)
    state.update(state_out)


    text_summary_chain = get_chain(selected_model, prompt)

    # 3.1 텍스트 요약 생성
    state_out = create_text_summary(text_summary_chain, state)
    state.update(state_out)
    
    trans_chain = get_translator(selected_model)
    state_out = create_text_trans_summary(trans_chain, state)
    state.update(state_out)
    

    # 3.2 Image 요약 생성 
    state_out = create_image_summary_data_batches(state)
    state.update(state_out)

    # 3.2.1 Image 요약 생성 
    state_out = create_image_summary(state)
    state.update(state_out)

    # 3.3 Table 요약 생성 
    state_out = create_table_summary_data_batches(state)
    state.update(state_out)

    # 3.3.1 Table 요약 생성 
    state_out = create_table_summary(state)
    state.update(state_out)

    # 3.4 Equation 요약 생성 
    state_out = create_equation_summary_data_batches(state)
    state.update(state_out)

    # 3.4.1 Equation 요약 생성 
    state_out = create_equation_summary(state)
    state.update(state_out)


    # 4 마크다운 표 생성 
    state_out = create_table_markdown(state)
    state.update(state_out)
    # 표를 마크다운 표로 다시 만들어?

    cnt = 1
    for key, value in state['equation_summary'].items():
        equation_html = f"<p id='{key}_1' data-category='equation' style='font-size:14px'>{value}</p>"
        state['html_content'].insert(cnt+key, equation_html)
        cnt+=1

    pdf_file = state["filepath"]  # PDF 파일 경로
    output_folder = os.path.splitext(pdf_file)[0]  # 출력 폴더 경로 설정
    filename = os.path.basename(pdf_file).split('.')[0]
    md_output_file = save_results(output_folder, filename, state['html_content'])

    output_file = '.'.join(file_path.split('.')[:-1]) + "_analy.json"             # pdf구조를 json으로 저장 
    with open(output_file, "w", encoding='utf-8') as file:
        json.dump(state, file, ensure_ascii=False)

    for del_file in state['split_filepaths'] + state['analyzed_files']:
        os.remove(del_file)
        
    return md_output_file, state


# 파일이 업로드 되었을 때
if uploaded_file:
    display_pdf(uploaded_file)
    md_output_file, state = main(uploaded_file, selected_model)

    # 마크다운 파일 읽기
    with open(md_output_file, "r", encoding="utf-8") as file:
        markdown_content = file.read()

    # 마크다운 내용 보여주기
    st_markdown(markdown_content)
    
    # vectordb 만들기 
    build_db(state)
    st.success(f"VectorDB를 생성하였습니다. 이제 논문에 대해 챗봇과 대화할 수 있습니다.")
    
    
if selected_paper:
    md_file = os.path.join(root_dir, selected_paper) + '/' + selected_paper + '.md'
    
    # 마크다운 파일 읽기
    with open(md_file, "r", encoding="utf-8") as file:
        markdown_content = file.read()
    
    st_markdown(markdown_content)