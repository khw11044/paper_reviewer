

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


analyzer = LayoutAnalyzer(os.environ.get("UPSTAGE_API_KEY"))
# analyzer = LayoutAnalyzer(UPSTAGE_API_KEY)


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


# 파일 업로드 전용 폴더: 임시로 저장 
root_dir = ".cache/files"
os.makedirs(root_dir, exist_ok=True)

st.set_page_config(
    page_title="현우의 논문 리뷰 AI Agent",
    page_icon=":star:"  # You can use emojis or provide a URL to a custom favicon image
)

# ------------------------------------- 구성: 제목 -------------------------------------------------
st.title("논문 읽기 웹 페이지입니다.")
st.markdown("원하는 논문을 선택하여 업로드 하면 요약, 번역, 챗봇까지 사용 가능합니다.")

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
# @st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")     # 스피너 작업중인 것을 표기해줌
def main1(file):
    
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f"./{root_dir}/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    
    # 1. 기본 문서 구조 분석
    state = GraphState(filepath=file_path, batch_size=10)
    state_out = split_pdf(state)
    state.update(state_out)
    my_bar.progress(1, text="문서 구조 분석 (1/20)")

    # # 1.1 문서 구조 분석기를 통해 기본 분석 결과 저장 
    state_out = analyze_layout(analyzer, state)
    state.update(state_out)
    my_bar.progress(2, text='..문서 구조 분석 (2/20)')

    # 1.2 문서에 대한 메타데이터 추출 
    state_out = extract_page_metadata(state)
    state.update(state_out)
    my_bar.progress(3, text='...문서 구조 분석 (3/20)')

    # 1.3 문서 구조와 내용에 대한 html 내용 추출
    # 페이지별 정보를 추출 
    state_out = extract_page_elements(state)
    state.update(state_out)
    my_bar.progress(4, text='....문서 정보 분석 (4/20)')

    # 1.4 문서 요소 별 tag 추출
    state_out = extract_tag_elements_per_page(state)
    state.update(state_out)
    my_bar.progress(5, text='.....문서 정보 분석 (5/20)')

    # 1.5 페이지 번호 추출 
    state_out = page_numbers(state)
    state.update(state_out)
    my_bar.progress(6, text='......문서 정보 분석 (6/20)')


    # 2.1 이미지를 추출하여 저장하고 위치를 저장 
    state_out = crop_image(state)
    state.update(state_out)
    my_bar.progress(7, text='........문서 구성 요소 분석 (7/20)')

    # 2.2 표를 추출하여 저장하고 위치를 저장 
    state_out = crop_table(state)
    state.update(state_out)
    my_bar.progress(8, text='..........문서 구성 요소 분석 (8/20)')

    # 2.3 수식을 추출하여 저장하고 위치를 저장 
    state_out = crop_equation(state)
    state.update(state_out)
    my_bar.progress(9, text='............문서 구성 요소 분석 (9/20)')

    # 2.4 텍스트를 추출하고 저장하여 위치를 저장 
    state_out = extract_page_text(state)
    state.update(state_out)
    my_bar.progress(10, text='..............문서 구성 요소 분석 (10/20)')
    
    pdf_file = state["filepath"]  # PDF 파일 경로
    output_folder = os.path.splitext(pdf_file)[0]  # 출력 폴더 경로 설정
    filename = os.path.basename(pdf_file).split('.')[0]
    
    md_output_file = save_results(output_folder, filename, state['html_content'])
    my_bar.progress(10, text='이제부터 요약 및 번역 작업을 수행하겠습니다 (10/20)')

    output_file = '.'.join(file_path.split('.')[:-1]) + "_analy.json"             # pdf구조를 json으로 저장 
    with open(output_file, "w", encoding='utf-8') as file:
        json.dump(state, file, ensure_ascii=False)

    
    my_bar.empty()
    return md_output_file, state



############################### 생성 ############################################
def main2(state, selected_model, file):
    my_bar = st.progress(0, text="업로드한 파일을 처리 중입니다...")
    file_path = f"./{root_dir}/{file.name}"
    
    from utils.prompt import summary_prompt, map_prompt, trans_prompt
    text_summary_chain = get_chain(selected_model, summary_prompt)
    paper_summary_chain = get_chain(selected_model, map_prompt)

    # 3.1 텍스트 요약 생성
    # 3.1.1 Map-reduce 방식 
    state_out = create_text_summary(text_summary_chain, state)
    state.update(state_out)
    
    state_out = map_reduce_summary(paper_summary_chain, state)
    state.update(state_out)
    my_bar.progress(11, text='................문서 요약 (11/20)')
    
    # 3.1.1 번역
    trans_chain = get_translator(selected_model, trans_prompt)
    state_out = create_text_trans_summary(trans_chain, state)
    state.update(state_out)
    my_bar.progress(12, text='..................문서 요약 번역 (12/20)')
    

    # 3.2 Image 요약 생성 
    state_out = create_image_summary_data_batches(state)
    state.update(state_out)
    my_bar.progress(13, text='....................문서 요약 (13/20)')

    # 3.2.1 Image 요약 생성 
    state_out = create_image_summary(state)
    state.update(state_out)
    my_bar.progress(14, text='......................문서 요약 (14/20)')

    # 3.3 Table 요약 생성 
    state_out = create_table_summary_data_batches(state)
    state.update(state_out)
    my_bar.progress(15, text='........................문서 요약 (15/20)')

    # 3.3.1 Table 요약 생성 
    state_out = create_table_summary(state)
    state.update(state_out)
    my_bar.progress(16, text='..........................문서 요약 (16/20)')

    # 3.4 Equation 요약 생성 
    state_out = create_equation_summary_data_batches(state)
    state.update(state_out)
    my_bar.progress(17, text='............................문서 요약 (17/20)')

    # 3.4.1 Equation 요약 생성 
    state_out = create_equation_summary(state)
    state.update(state_out)
    my_bar.progress(18, text='..............................문서 요약 (18/20)')


    # 4 마크다운 표 생성 
    state_out = create_table_markdown(state)
    state.update(state_out)
    my_bar.progress(19, text='................................표 생성 (19/20)')
    # 표를 마크다운 표로 다시 만들어?

    cnt = 1
    for key, value in state['equation_summary'].items():
        equation_html = f"<p id='{key}_1' data-category='equation' style='font-size:14px'>{value}</p>"
        state['html_content'].insert(cnt+int(key), equation_html)
        cnt+=1

    pdf_file = state["filepath"]  # PDF 파일 경로
    output_folder = os.path.splitext(pdf_file)[0]  # 출력 폴더 경로 설정
    filename = os.path.basename(pdf_file).split('.')[0]
    md_output_file = save_results(output_folder, filename, state['html_content'])
    my_bar.progress(20, text='문서 분석 내용 저장 (20/20)')

    output_file = '.'.join(file_path.split('.')[:-1]) + "_analy.json"             # pdf구조를 json으로 저장 
    with open(output_file, "w", encoding='utf-8') as file:
        json.dump(state, file, ensure_ascii=False)

    for del_file in state['split_filepaths'] + state['analyzed_files']:
        os.remove(del_file)
    
    my_bar.empty()
    return md_output_file, state

# 파일이 업로드 되었을 때
if uploaded_file:
    my_bar = st.progress(0, text="업로드한 파일을 처리 중입니다...")
    md_output_file, state = main1(uploaded_file)

    # 마크다운 파일 읽기
    with open(md_output_file, "r", encoding="utf-8") as file:
        markdown_content = file.read()

    # 마크다운 내용 보여주기
    st_markdown(markdown_content)
    
    md_output_file, state = main2(state, selected_model, uploaded_file)
    
    
    # vectordb 만들기 
    build_db(state)
    st.success(f"VectorDB를 생성하였습니다. 이제 논문에 대해 챗봇과 대화할 수 있습니다.")
    
    
if selected_paper:
    md_file = os.path.join(root_dir, selected_paper) + '/' + selected_paper + '.md'
    
    # 마크다운 파일 읽기
    with open(md_file, "r", encoding="utf-8") as file:
        markdown_content = file.read()
    
    st_markdown(markdown_content)
