import streamlit as st
import json 
import os 
import glob
from langchain.schema import Document
from utils.funcs import html_to_markdown_table
from utils.creates import get_translator
import re
from utils.prompt import trans_prompt

# OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
# UPSTAGE_API_KEY = st.secrets["UPSTAGE_API_KEY"]
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# os.environ["UPSTAGE_API_KEY"] = UPSTAGE_API_KEY

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

# ------------------------------------- 구성: 제목 -------------------------------------------------
st.title("논문 번역 요약 읽기")

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


if selected_paper:
    markdown_file_path = f'{root_dir}/{selected_paper}_translation.md'
    
    if os.path.exists(markdown_file_path):
        print('파일 이미 있음, 불러오기')
        # 마크다운 파일 읽기
        with open(markdown_file_path, "r", encoding="utf-8") as file:
            markdown_content = file.read()
        
        st_markdown(markdown_content)
    
    else:
        json_file = root_dir + '/' + selected_paper +'_analy.json'
        
        print('json_file:',json_file)

        with open(json_file, "r", encoding='utf-8') as f:
            json_data = json.load(f)

        markdown_contents = []  # 마크다운 내용을 저장할 빈 리스트

        names = json_data['section_names']
        # 번역기 text는 이미 번역이 되어있고 이미지에 대한 설명부분을 번역하기 위헤 
        trans_chain = get_translator(selected_model, trans_prompt)
        
        
        for i, page in enumerate(json_data['section_elements'].keys()):
            page = int(page)
            print(names[page])
            if i == 0:
                text_summary = json_data['paper_trans_summary']
            else:
                text_summary = json_data['texts_trans_summary'][str(page)]
                
            section_title = f'# {names[page]}'
            st.markdown(section_title)
            st.markdown(text_summary)
            
            markdown_contents.append(section_title)
            markdown_contents.append(text_summary)
            
            for image_summary_data_batch in json_data['image_summary_data_batches']:
                if image_summary_data_batch['page'] == page:
                    img_file = image_summary_data_batch['image']
                    st.image(img_file)
                    
                    img_name = os.path.basename(img_file).split('.')[0]
                    
                    markdown_result = html_to_markdown_table(json_data['images_summary'][img_name])
                    doc = Document(page_content=markdown_result)
                    markdown_result = trans_chain.invoke([doc])
                    st.markdown(markdown_result, unsafe_allow_html=True)
                    
                    # 이미지와 테이블 마크다운을 리스트에 추가
                    markdown_contents.append(f'![{img_name}]({img_file})')
                    markdown_contents.append(markdown_result)
                    
                    
            for table_summary_data_batch in json_data['table_summary_data_batches']:
                if table_summary_data_batch['page'] == page:
                    table_img_file = table_summary_data_batch['table']
                    table_text = table_summary_data_batch['text']
                    st.image(table_img_file)
                    
                    table_img_name = os.path.basename(table_img_file).split('.')[0]
                    markdown_result = html_to_markdown_table(json_data['tables_summary'][table_img_name])
                    doc = Document(page_content=markdown_result)
                    markdown_result = trans_chain.invoke([doc])
                    st.markdown(markdown_result, unsafe_allow_html=True)
                    
                    # 테이블과 텍스트도 리스트에 추가
                    markdown_contents.append(f'![{table_img_name}]({table_img_file})')
                    markdown_contents.append(markdown_result)


        # 리스트에 저장된 마크다운 내용을 하나의 파일로 저장
        with open(markdown_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(markdown_contents))

        st.success(f"마크다운 파일을 '{markdown_file_path}'에 저장하였습니다.")
