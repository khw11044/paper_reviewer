import pymupdf
import os 

from utils.Classes import GraphState

def page_numbers(state: GraphState):
    return GraphState(page_numbers=list(state["section_elements"].keys()))


def split_pdf(state: GraphState):
    """
    입력 PDF를 여러 개의 작은 PDF 파일로 분할합니다.

    :param state: GraphState 객체, PDF 파일 경로와 배치 크기 정보를 포함
    :return: 분할된 PDF 파일 경로 목록을 포함한 GraphState 객체
    """
    # PDF 파일 경로와 배치 크기 추출
    filepath = state["filepath"]
    batch_size = state["batch_size"]

    # PDF 파일 열기
    input_pdf = pymupdf.open(filepath)
    num_pages = len(input_pdf)
    print(f"총 페이지 수: {num_pages}")

    ret = []
    # PDF 분할 작업 시작
    for start_page in range(0, num_pages, batch_size):
        # 배치의 마지막 페이지 계산 (전체 페이지 수를 초과하지 않도록)
        end_page = min(start_page + batch_size, num_pages) - 1

        # 분할된 PDF 파일명 생성
        input_file_basename = os.path.splitext(filepath)[0]
        output_file = f"{input_file_basename}_{start_page:04d}_{end_page:04d}.pdf"
        print(f"분할 PDF 생성: {output_file}")

        # 새로운 PDF 파일 생성 및 페이지 삽입
        with pymupdf.open() as output_pdf:
            output_pdf.insert_pdf(input_pdf, from_page=start_page, to_page=end_page)
            output_pdf.save(output_file)
            ret.append(output_file)

    # 원본 PDF 파일 닫기
    input_pdf.close()

    # 분할된 PDF 파일 경로 목록을 포함한 GraphState 객체 반환
    return GraphState(split_filepaths=ret)


# 분할된 PDF 파일 목록을 가져옵니다.
def analyze_layout(analyzer, state: GraphState):

    split_files = state["split_filepaths"]

    # 분석된 파일들의 경로를 저장할 리스트를 초기화합니다.
    analyzed_files = []

    # 각 분할된 PDF 파일에 대해 레이아웃 분석을 수행합니다.
    for file in split_files:
        # 레이아웃 분석을 실행하고 결과 파일 경로를 리스트에 추가합니다.
        analyzed_files.append(analyzer.execute(file))

    # image_processor = PDFImageProcessor(file_path)
    # output_folder, filename, html_content = image_processor.extract_images()
    
    # md_output_file = save_results(output_folder, filename, html_content)
    
    # 분석된 파일 경로들을 정렬하여 새로운 GraphState 객체를 생성하고 반환합니다.
    # 정렬은 파일들의 순서를 유지하기 위해 수행됩니다.
    return GraphState(analyzed_files=sorted(analyzed_files))


from bs4 import BeautifulSoup
def html_to_markdown_table(html_content):
    # BeautifulSoup으로 HTML 파싱
    soup = BeautifulSoup(html_content, "html.parser")
    
    # 제목 추출
    title = soup.find("title").text if soup.find("title") else "No title"
    
    # 요약 추출
    summary = soup.find("summary").text if soup.find("summary") else "No summary"
    
    # 엔터티 추출
    entities = []
    entity_tags = soup.find_all("entity")
    for entity in entity_tags:
        entities.append(entity.text.strip())
    
    # 마크다운 표 생성
    markdown_table = "| Title | Summary | Entities |\n"
    markdown_table += "|-------|---------|----------|\n"
    markdown_table += f"| {title} | {summary} | {', '.join(entities)} |\n"
    
    return markdown_table