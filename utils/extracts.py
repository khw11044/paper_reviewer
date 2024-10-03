import os 
import re
import json
from bs4 import BeautifulSoup

from utils.Classes import GraphState

def convert_to_list(html):
    soup = BeautifulSoup(html, "html.parser")

    # <p> 태그를 찾고 그 안의 내용을 처리
    p_tag = soup.find("p", {"data-category": "list"})
    
    if p_tag:
        # '•' 기호를 기준으로 텍스트를 분리
        items = re.split(r'(•)', p_tag.text)

        # 새로운 <ul> 태그 생성
        ul_tag = soup.new_tag("ul")

        # <p> 태그의 id, data-category, style 속성을 <ul> 태그로 옮김
        if p_tag.has_attr('id'):
            ul_tag['id'] = p_tag['id']
        if p_tag.has_attr('data-category'):
            ul_tag['data-category'] = p_tag['data-category']
        if p_tag.has_attr('style'):
            ul_tag['style'] = p_tag['style']

        current_item = ""
        for item in items:
            if item == '•':  # '•' 패턴인 경우
                if current_item.strip():  # 빈 항목은 제외
                    li_tag = soup.new_tag("li")
                    li_tag.string = current_item.strip()
                    ul_tag.append(li_tag)
                current_item = item  # 새로운 리스트 항목 시작
            else:
                current_item += item

        # 마지막 리스트 항목 추가
        if current_item.strip():
            li_tag = soup.new_tag("li")
            li_tag.string = current_item.strip()
            ul_tag.append(li_tag)

        # <p> 태그를 <ul> 태그로 교체
        p_tag.replace_with(ul_tag)

    return str(soup)

def convert_to_numbered_list(html):
    soup = BeautifulSoup(html, "html.parser")

    # <p> 태그를 찾고 그 안의 내용을 처리
    p_tag = soup.find("p", {"data-category": "list"})
    
    if p_tag:
        # '[숫자]' 패턴을 기준으로 텍스트를 분리
        items = re.split(r'(\[\d+\])', p_tag.text)

        # 새로운 <ul> 태그 생성
        ul_tag = soup.new_tag("ul")

        # <p> 태그의 id, data-category, style 속성을 <ul> 태그로 옮김
        if p_tag.has_attr('id'):
            ul_tag['id'] = p_tag['id']
        if p_tag.has_attr('data-category'):
            ul_tag['data-category'] = p_tag['data-category']
        if p_tag.has_attr('style'):
            ul_tag['style'] = p_tag['style']

        current_item = ""
        for item in items:
            if re.match(r'\[\d+\]', item):  # '[숫자]' 패턴인 경우
                if current_item.strip():  # 빈 항목은 제외
                    li_tag = soup.new_tag("li")
                    li_tag.string = current_item.strip()
                    ul_tag.append(li_tag)
                current_item = item  # 새로운 숫자 리스트 시작
            else:
                current_item += item

        # 마지막 리스트 항목 추가
        if current_item.strip():
            li_tag = soup.new_tag("li")
            li_tag.string = current_item.strip()
            ul_tag.append(li_tag)

        # <p> 태그를 <ul> 태그로 교체
        p_tag.replace_with(ul_tag)

    return str(soup)

def process_html(html):
    if '•' in html:
        return convert_to_list(html)  # •가 있으면 convert_to_list 함수 호출
    elif re.search(r'\[\d+\]', html):
        return convert_to_numbered_list(html)  # [숫자]가 있으면 convert_to_numbered_list 함수 호출
    else:
        return html  # 아무 조건도 맞지 않으면 원본 html 반환


def process_html_string(html_str):
    # 1. 'para-<br>metric'처럼 <br> 앞에 '-'가 있으면 붙여준다.
    html_str = re.sub(r'-<br>', '', html_str)

    # 2. '<br>'을 만나면 한 칸 띄어쓰기
    html_str = re.sub(r'<br>', ' ', html_str)

    return html_str

def extract_start_end_page(filename):
    """
    파일 이름에서 시작 페이지와 끝 페이지 번호를 추출하는 함수입니다.

    :param filename: 분석할 파일의 이름
    :return: 시작 페이지 번호와 끝 페이지 번호를 튜플로 반환
    """
    # 파일 경로에서 파일 이름만 추출
    file_name = os.path.basename(filename)
    # 파일 이름을 '_' 기준으로 분리
    file_name_parts = file_name.split("_")

    if len(file_name_parts) >= 3:
        # 파일 이름의 뒤에서 두 번째 부분에서 숫자를 추출하여 시작 페이지로 설정
        start_page = int(re.findall(r"(\d+)", file_name_parts[-2])[0])
        # 파일 이름의 마지막 부분에서 숫자를 추출하여 끝 페이지로 설정
        end_page = int(re.findall(r"(\d+)", file_name_parts[-1])[0])
    else:
        # 파일 이름 형식이 예상과 다를 경우 기본값 설정
        start_page, end_page = 0, 0

    return start_page, end_page

def extract_page_metadata(state: GraphState):
    """
    분석된 JSON 파일들에서 페이지 메타데이터를 추출하는 함수입니다.

    :param state: 현재의 GraphState 객체
    :return: 페이지 메타데이터가 추가된 새로운 GraphState 객체
    """
    # 분석된 JSON 파일 목록 가져오기
    json_files = state["analyzed_files"]

    # 페이지 메타데이터를 저장할 딕셔너리 초기화
    page_metadata = dict()

    for json_file in json_files:
        # JSON 파일 열기 및 데이터 로드
        with open(json_file, "r", encoding='utf-8') as f:
            data = json.load(f)

        # 파일명에서 시작 페이지 번호 추출
        start_page, _ = extract_start_end_page(json_file)

        # JSON 데이터에서 각 페이지의 메타데이터 추출
        for element in data["metadata"]["pages"]:
            # 원본 페이지 번호
            original_page = int(element["page"])
            # 상대적 페이지 번호 계산 (전체 문서 기준)
            relative_page = start_page + original_page - 1

            # 페이지 크기 정보 추출
            metadata = {
                "size": [
                    int(element["width"]),
                    int(element["height"]),
                ],
            }
            # 상대적 페이지 번호를 키로 하여 메타데이터 저장
            page_metadata[relative_page] = metadata

    # 추출된 페이지 메타데이터로 새로운 GraphState 객체 생성 및 반환
    return GraphState(page_metadata=page_metadata)


def extract_page_elements(state: GraphState):
    # 분석된 JSON 파일 목록을 가져옵니다.
    json_files = state["analyzed_files"]
    
    pdf_file = state["filepath"]  # PDF 파일 경로
    output_folder = os.path.splitext(pdf_file)[0]  # 출력 폴더 경로 설정
    filename = os.path.basename(pdf_file).split('.')[0]
    
    # 섹션별 요소를 저장할 딕셔너리를 초기화합니다.
    section_elements = dict()
    section_names = []
    html_content = []

    # 전체 문서에서 고유한 요소 ID를 부여하기 위한 카운터입니다.
    element_id = 0
    section_id = -1
    section_name =''

    # 각 JSON 파일을 순회하며 처리합니다.
    for json_file in json_files:
        # 파일명에서 시작 페이지 번호를 추출합니다.
        start_page, _ = extract_start_end_page(json_file)

        # JSON 파일을 열어 데이터를 로드합니다.
        with open(json_file, "r", encoding='utf-8') as f:
            data = json.load(f)

        # JSON 데이터의 각 요소를 처리합니다.
        for element in data["elements"]:
        
            soup = BeautifulSoup(element["html"], "html.parser")
            if element["category"] == "paragraph":
                subtitle_tag = soup.find(["p", "h1"])
                
                # 1.1. 이런식인 경우 중 제목
                if subtitle_tag.string and re.match(r"^\d+\.\d+\.", subtitle_tag.string.strip()):       # Sub section 인 경우
                    new_subtitle_tag = soup.new_tag("h2", id=subtitle_tag.get("id"))
                    if subtitle_tag.string:
                        new_subtitle_tag.string = subtitle_tag.string
                        subtitle_tag.replace_with(new_subtitle_tag)
                        
                        element["category"] = 'heading2'
                        element["html"] = str(soup)
                        
                elif subtitle_tag.string and re.match(r"^\d+\.\d+\.\d+\.", subtitle_tag.string.strip()):  # Sub sub section 인 경우
                    new_subtitle_tag = soup.new_tag("h3", id=subtitle_tag.get("id"))
                    if subtitle_tag.string:
                        new_subtitle_tag.string = subtitle_tag.string
                        subtitle_tag.replace_with(new_subtitle_tag)

                        element["category"] = 'heading3'
                        element["html"] = str(soup)
                        
                else:
                    subtitle_tag = soup.find("p", {"style": "font-size:20px"})
                    if subtitle_tag:
                        new_subtitle_tag = soup.new_tag("h1", id=subtitle_tag.get("id"))
                        if subtitle_tag.string:
                            new_subtitle_tag.string = subtitle_tag.string
                            subtitle_tag.replace_with(new_subtitle_tag)
                        
                            element["category"] = 'heading1'
                            element["html"] = str(soup)
                    
                    else:
                        element["html"] = process_html_string(element["html"])

                        
            elif element["category"] == "caption":
                caption_tag = soup.find("caption")
                    
                if caption_tag:
                    new_caption_tag = soup.new_tag("blockquote", id=caption_tag.get("id"), style=caption_tag["style"])
                    if caption_tag.string:
                        new_caption_tag.string = caption_tag.string
                        caption_tag.replace_with(new_caption_tag)
                        element["html"] = str(soup)
            
            elif element["category"] == "figure":
                img_tag = soup.find("img")
                img_tag["alt"] = img_tag["alt"].replace('-\n', '').replace('\n', ' ')
                img_tag["src"] = os.path.join(output_folder, str(element['id'])+'.png').replace("\\", "/")
                element["html"] = str(soup)
            
            elif element["category"] == "equation":
                equation_tag = soup.find("p", {"data-category": "equation"})
                equation_img_tag = soup.new_tag("img")
                equation_img_tag["alt"] = "equation"
                equation_img_tag["id"] = str(element['id'])
                equation_img_tag["src"] = os.path.join(output_folder, str(element['id'])+'.png').replace("\\", "/")
                equation_tag.insert_after(equation_img_tag)
                equation_tag.decompose()  # 기존 수식 태그를 삭제
                element["html"] = str(soup)
                
                
            elif element["category"] == "list":
                element["html"] = process_html_string(element["html"])
                element["html"] = process_html(element["html"])
                    
                
            if element["category"] == 'heading1':
                section_name = soup.find('h1').get_text(separator="\n")
                section_name = section_name.replace('-\n', '').replace('\n', ' ')
                section_names.append(section_name)
                section_id += 1
                element["html"] = process_html_string(element["html"])
            
            # 원본 페이지 번호를 정수로 변환합니다.
            original_page = int(element["page"])
            # 전체 문서 기준의 상대적 페이지 번호를 계산합니다.
            relative_page = start_page + original_page - 1

                
            if section_id not in section_elements:
                section_elements[section_id] = []

            # 요소에 고유 ID를 부여합니다.
            element["id"] = element_id
            element_id += 1

            # 요소의 페이지 번호를 상대적 페이지 번호로 업데이트합니다.
            element["page"] = relative_page
            # 요소를 해당 페이지의 리스트에 추가합니다.
            
            if element["category"] != "footnote":
                section_elements[section_id].append(element)
                html_content.append(element["html"])
            
    # 최종적으로 html_content가 md가 된다. 
    # 추출된 페이지별 요소 정보로 새로운 GraphState 객체를 생성하여 반환합니다.
    return GraphState(section_elements=section_elements, section_names=section_names, html_content=html_content)



def extract_tag_elements_per_page(state: GraphState):
    # GraphState 객체에서 페이지 요소들을 가져옵니다.
    section_elements = state["section_elements"]

    # 파싱된 페이지 요소들을 저장할 새로운 딕셔너리를 생성합니다.
    parsed_section_elements = dict()

    # 각 페이지와 해당 페이지의 요소들을 순회합니다.
    for key, section_element in section_elements.items():
        # 이미지, 테이블, 텍스트 요소들을 저장할 리스트를 초기화합니다.
        text_elements = []
        image_elements = []
        table_elements = []
        equation_elements = []

        # 페이지의 각 요소를 순회하며 카테고리별로 분류합니다.
        for element in section_element:
            if element["category"] == "figure":
                # 이미지 요소인 경우 image_elements 리스트에 추가합니다.
                image_elements.append(element)
            elif element["category"] == "table":
                # 테이블 요소인 경우 table_elements 리스트에 추가합니다.
                table_elements.append(element)
            elif element["category"] == "equation":
                # 테이블 요소인 경우 table_elements 리스트에 추가합니다.
                equation_elements.append(element)
            else:
                # 그 외의 요소는 모두 텍스트 요소로 간주하여 text_elements 리스트에 추가합니다.
                text_elements.append(element)

        # 분류된 요소들을 페이지 키와 함께 새로운 딕셔너리에 저장합니다.
        parsed_section_elements[key] = {
            "text_elements": text_elements,
            "image_elements": image_elements,
            "table_elements": table_elements,
            "equation_elements": equation_elements,
            "elements": section_element,  # 원본 페이지 요소도 함께 저장합니다.
        }

    # 파싱된 페이지 요소들을 포함한 새로운 GraphState 객체를 반환합니다.
    return GraphState(section_elements=parsed_section_elements)



def extract_page_text(state: GraphState):
    # 상태 객체에서 페이지 번호 목록을 가져옵니다.
    section_names = state["section_elements"]

    # 추출된 텍스트를 저장할 딕셔너리를 초기화합니다.
    extracted_texts = dict()

    # 각 페이지 번호에 대해 반복합니다.
    for section_name in section_names:
        # 현재 페이지의 텍스트를 저장할 빈 문자열을 초기화합니다.
        extracted_texts[section_name] = ""

        # 현재 페이지의 모든 텍스트 요소에 대해 반복합니다.
        for i, element in enumerate(state["section_elements"][section_name]["text_elements"]):
            # 각 텍스트 요소의 내용을 현재 페이지의 텍스트에 추가합니다.
            if i == 0:
                extracted_texts[section_name] += element["text"] +' \n '
            else:
                extracted_texts[section_name] += element["text"] 

    # 추출된 텍스트를 포함한 새로운 GraphState 객체를 반환합니다.
    return GraphState(texts=extracted_texts)


