import os 


from bs4 import BeautifulSoup
from utils.Classes import GraphState, ImageCropper

def crop_equation(state: GraphState):
    """
    PDF 파일에서 이미지를 추출하고 크롭하는 함수

    :param state: GraphState 객체
    :return: 크롭된 이미지 정보가 포함된 GraphState 객체
    """
    pdf_file = state["filepath"]  # PDF 파일 경로
    section_numbers = state["section_elements"]  # 처리할 페이지 번호 목록
    output_folder = os.path.splitext(pdf_file)[0]  # 출력 폴더 경로 설정
    os.makedirs(output_folder, exist_ok=True)  # 출력 폴더 생성

    cropped_equations = dict()  # 크롭된 이미지 정보를 저장할 딕셔너리
    for section_num in section_numbers:
        cur_page_num = -1
        for element in state["section_elements"][section_num]["equation_elements"]:
            page_num = element['page']
            if cur_page_num != page_num:
                cur_page_num = page_num
                pdf_image = ImageCropper.pdf_to_image(pdf_file, page_num)  # PDF 페이지를 이미지로 변환
        
            if element["category"] == "equation":
                # 이미지 요소의 좌표를 정규화
                normalized_coordinates = ImageCropper.normalize_coordinates(
                    element["bounding_box"], state["page_metadata"][page_num]["size"]
                )

                # 크롭된 이미지 저장 경로 설정
                output_file = os.path.join(output_folder, f"{element['id']}.png")
                # 이미지 크롭 및 저장
                ImageCropper.crop_image(pdf_image, normalized_coordinates, output_file)
                cropped_equations[element["id"]] = output_file
                print(f"page:{page_num}, id:{element['id']}, path: {output_file}")
                
            
    return GraphState(equation=cropped_equations)  # 크롭된 이미지 정보를 포함한 GraphState 반환


def crop_image(state: GraphState):
    """
    PDF 파일에서 이미지를 추출하고 크롭하는 함수

    :param state: GraphState 객체
    :return: 크롭된 이미지 정보가 포함된 GraphState 객체
    """
    pdf_file = state["filepath"]  # PDF 파일 경로
    section_numbers = state["section_elements"]  # 처리할 페이지 번호 목록
    output_folder = os.path.splitext(pdf_file)[0]  # 출력 폴더 경로 설정
    os.makedirs(output_folder, exist_ok=True)  # 출력 폴더 생성

    cropped_images = dict()  # 크롭된 이미지 정보를 저장할 딕셔너리
    for section_num in section_numbers:
        cur_page_num = -1
        for element in state["section_elements"][section_num]["image_elements"]:
            page_num = element['page']
            if cur_page_num != page_num:
                cur_page_num = page_num
                pdf_image = ImageCropper.pdf_to_image(pdf_file, page_num)  # PDF 페이지를 이미지로 변환
        
            if element["category"] == "figure":
                # 이미지 요소의 좌표를 정규화
                normalized_coordinates = ImageCropper.normalize_coordinates(
                    element["bounding_box"], state["page_metadata"][page_num]["size"]
                )

                # 크롭된 이미지 저장 경로 설정
                output_file = os.path.join(output_folder, f"{element['id']}.png")
                # 이미지 크롭 및 저장
                ImageCropper.crop_image(pdf_image, normalized_coordinates, output_file)
                cropped_images[element["id"]] = output_file
                print(f"page:{page_num}, id:{element['id']}, path: {output_file}")
                
            
    return GraphState(images=cropped_images)  # 크롭된 이미지 정보를 포함한 GraphState 반환


def crop_table(state: GraphState):
    """
    PDF 파일에서 표를 추출하고 크롭하는 함수

    :param state: GraphState 객체
    :return: 크롭된 표 이미지 정보가 포함된 GraphState 객체
    """
    pdf_file = state["filepath"]  # PDF 파일 경로
    section_numbers = state["section_elements"]  # 처리할 페이지 번호 목록
    output_folder = os.path.splitext(pdf_file)[0]  # 출력 폴더 경로 설정
    os.makedirs(output_folder, exist_ok=True)  # 출력 폴더 생성

    cropped_images = dict()  # 크롭된 표 이미지 정보를 저장할 딕셔너리
    for section_num in section_numbers:
        cur_page_num = -1
        for element in state["section_elements"][section_num]["table_elements"]:
            page_num = element['page']
            
            if cur_page_num != page_num:
                cur_page_num = page_num
                pdf_image = ImageCropper.pdf_to_image(pdf_file, page_num)  # PDF 페이지를 이미지로 변환
        
            if element["category"] == "table":
                # 표 요소의 좌표를 정규화
                normalized_coordinates = ImageCropper.normalize_coordinates(
                    element["bounding_box"], state["page_metadata"][page_num]["size"]
                )

                # 크롭된 표 이미지 저장 경로 설정
                output_file = os.path.join(output_folder, f"{element['id']}.png")
                # 표 이미지 크롭 및 저장
                ImageCropper.crop_image(pdf_image, normalized_coordinates, output_file)
                cropped_images[element["id"]] = output_file
                print(f"page:{page_num}, id:{element['id']}, path: {output_file}")
    return GraphState(
        tables=cropped_images
    )  # 크롭된 표 이미지 정보를 포함한 GraphState 반환




