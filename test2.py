

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


# 파일 업로드
uploaded_file = root_dir + '/example.pdf'

# 모델 선택 메뉴
selected_model = "gpt-4o-mini"



# 파일이 업로드 되었을 때
# 파일을 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
# @st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")     # 스피너 작업중인 것을 표기해줌
def main1(file_path):
    
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
    

    # 2.3 수식을 추출하여 저장하고 위치를 저장 
    state_out = crop_equation(state)
    state.update(state_out)

    # 2.4 텍스트를 추출하고 저장하여 위치를 저장 
    state_out = extract_page_text(state)
    state.update(state_out)

    
    pdf_file = state["filepath"]  # PDF 파일 경로
    output_folder = os.path.splitext(pdf_file)[0]  # 출력 폴더 경로 설정
    filename = os.path.basename(pdf_file).split('.')[0]
    
    md_output_file = save_results(output_folder, filename, state['html_content'])

    output_file = '.'.join(file_path.split('.')[:-1]) + "_analy.json"             # pdf구조를 json으로 저장 
    with open(output_file, "w", encoding='utf-8') as file:
        json.dump(state, file, ensure_ascii=False)

    
    return md_output_file, state


############################### 생성 ############################################
def main2(state, selected_model, file_path):

    # file_path = f"./{root_dir}/{file.name}"
    
    from utils.prompt import summary_prompt, map_prompt, trans_prompt
    text_summary_chain = get_chain(selected_model, summary_prompt)
    paper_summary_chain = get_chain(selected_model, map_prompt)

    # 3.1 텍스트 요약 생성
    # 3.1.1 Map-reduce 방식 
    state_out = create_text_summary(text_summary_chain, state)
    state.update(state_out)
    
    state_out = map_reduce_summary(paper_summary_chain, state)
    state.update(state_out)

    
    # 3.1.1 번역
    trans_chain = get_translator(selected_model, trans_prompt)
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
    # 수식 넣기 
    cnt = 1
    for key, value in state['equation_summary'].items():
        equation_html = f"<p id='{key}_1' data-category='equation' style='font-size:14px'>{value}</p>"
        state['html_content'].insert(cnt+int(key), equation_html)
        cnt+=1

    pdf_file = state["filepath"]  # PDF 파일 경로
    output_folder = os.path.splitext(pdf_file)[0]  # 출력 폴더 경로 설정
    filename = os.path.basename(pdf_file).split('.')[0]
    md_output_file = save_results(output_folder, filename, state['html_content'])
    
    output_file = '.'.join(file_path.split('.')[:-1]) + "_analy.json"             # pdf구조를 json으로 저장 
    with open(output_file, "w", encoding='utf-8') as file:
        json.dump(state, file, ensure_ascii=False)

    for del_file in state['split_filepaths'] + state['analyzed_files']:
        if os.path.isdir(del_file):
            os.remove(del_file)
    
    return md_output_file, state
    
# 파일이 업로드 되었을 때
if uploaded_file:

    md_output_file, state = main1(uploaded_file)

    # md_output_file = './.cache/files/example/example.md'
    
    # json_file = './.cache/files/example_analy.json'
    
    # with open(json_file, "r", encoding='utf-8') as f:
    #     state_load = json.load(f)
    
    # 마크다운 파일 읽기
    with open(md_output_file, "r", encoding="utf-8") as file:
        markdown_content = file.read()

    
    _, state = main2(state, selected_model, uploaded_file)
    
    
    # vectordb 만들기 
    build_db(state)
    st.success(f"VectorDB를 생성하였습니다. 이제 논문에 대해 챗봇과 대화할 수 있습니다.")
    
    
