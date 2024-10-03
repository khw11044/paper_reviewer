from utils.Classes import GraphState
from langchain_core.documents import Document

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.documents import Document



def get_chain(model_name, prompt):
    # ChatOpenAI 모델의 또 다른 인스턴스를 생성합니다. (이전 인스턴스와 동일한 설정)
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=0,
    )

    # 문서 요약을 위한 체인을 생성합니다.
    # 이 체인은 여러 문서를 입력받아 하나의 요약된 텍스트로 결합합니다.
    prompt = PromptTemplate.from_template(prompt)
    text_summary_chain = create_stuff_documents_chain(llm, prompt)
    
    return text_summary_chain


def create_text_summary(text_summary_chain, state: GraphState):
    # state에서 텍스트 데이터를 가져옵니다.
    texts = state["texts"]

    # 요약된 텍스트를 저장할 딕셔너리를 초기화합니다.
    text_summary = dict()

    # texts.items()를 페이지 번호(키)를 기준으로 오름차순 정렬합니다.
    sorted_texts = sorted(texts.items(), key=lambda x: x[0])

    # 각 페이지의 텍스트를 Document 객체로 변환하여 입력 리스트를 생성합니다.
    inputs = [
        {"context": [Document(page_content=text)]} for page_num, text in sorted_texts
    ]

    # text_summary_chain을 사용하여 일괄 처리로 요약을 생성합니다.
    summaries = text_summary_chain.batch(inputs)

    # 생성된 요약을 페이지 번호와 함께 딕셔너리에 저장합니다.
    for page_num, summary in enumerate(summaries):
        text_summary[page_num] = summary

    # 요약된 텍스트를 포함한 새로운 GraphState 객체를 반환합니다.
    return GraphState(text_summary=text_summary)

# def map_reduce_summary(paper_summary_chain, state: GraphState):
#     # state에서 텍스트 데이터를 가져옵니다.
#     texts = state["text_summary"]

#     # 요약된 텍스트를 저장할 딕셔너리를 초기화합니다.
#     paper_summary = dict()

#     # 각 페이지의 텍스트를 Document 객체로 변환하여 입력 리스트를 생성합니다.
#     inputs = [
#         {"context": [Document(page_content=text)]} for page_num, text in texts.items()
#     ]

#     # paper_summary_chain을 사용하여 일괄 처리로 요약을 생성합니다.
#     summaries = paper_summary_chain.batch(inputs)

#     # 생성된 요약을 페이지 번호와 함께 딕셔너리에 저장합니다.
#     for page_num, summary in enumerate(summaries):
#         paper_summary[page_num] = summary

#     # 요약된 텍스트를 포함한 새로운 GraphState 객체를 반환합니다.
#     return GraphState(paper_summary=paper_summary)


def map_reduce_summary(paper_summary_chain, state: GraphState):
    # state에서 텍스트 데이터를 가져옵니다.
    texts = state["text_summary"]

    inputs = [text for _, text in texts.items()]
    inputs = {"context": [Document(page_content="\n".join(inputs))]}

    # paper_summary_chain을 사용하여 일괄 처리로 요약을 생성합니다.
    summaries = paper_summary_chain.invoke(inputs)

    # 요약된 텍스트를 포함한 새로운 GraphState 객체를 반환합니다.
    return GraphState(paper_summary=summaries)


def create_text_trans_summary(trans_chain, state: GraphState):

    texts = state["text_summary"]
    
    text_trans_summary = dict()
    
    # texts.items()를 페이지 번호(키)를 기준으로 오름차순 정렬합니다.
    sorted_texts = sorted(texts.items(), key=lambda x: x[0])

    # 각 페이지의 텍스트를 Document 객체로 변환하여 입력 리스트를 생성합니다.
    inputs = [
        {"context": [Document(page_content=text)]} for page_num, text in sorted_texts
    ]

    # text_summary_chain을 사용하여 일괄 처리로 요약을 생성합니다.
    summaries = trans_chain.batch(inputs)

    # 생성된 요약을 페이지 번호와 함께 딕셔너리에 저장합니다.
    for page_num, summary in enumerate(summaries):
        text_trans_summary[page_num] = summary

    paper_summary = state["paper_summary"]
    # inputs = [text for _, text in paper_summary.items()]
    # summaries_trans = trans_chain.invoke(inputs)
    summaries_trans = trans_chain.invoke({"context": [Document(page_content=paper_summary)]})
    # 요약된 텍스트를 포함한 새로운 GraphState 객체를 반환합니다.
    return GraphState(texts_trans_summary=text_trans_summary, paper_trans_summary=summaries_trans)


def create_table_markdown(table_markdown_extractor, state: GraphState):
    # table_markdown_extractor를 사용하여 테이블 마크다운 생성
    # state["table_summary_data_batches"]에 저장된 테이블 데이터를 사용
    table_markdowns = table_markdown_extractor.invoke(
        state["table_summary_data_batches"],
    )

    # 결과를 저장할 딕셔너리 초기화
    table_markdown_output = dict()

    # 각 데이터 배치와 생성된 테이블 마크다운을 매칭하여 저장
    for data_batch, table_summary in zip(
        state["table_summary_data_batches"], table_markdowns
    ):
        # 데이터 배치의 id를 키로 사용하여 테이블 마크다운 저장
        table_markdown_output[data_batch["id"]] = table_summary

    # 새로운 GraphState 객체 반환, table_markdown 키에 결과 저장
    return GraphState(table_markdown=table_markdown_output)

def create_equation_summary_data_batches(state: GraphState):
    # 이미지 요약을 위한 데이터 배치를 생성하는 함수
    data_batches = []

    # 페이지 번호를 오름차순으로 정렬
    page_numbers = sorted(list(state["section_elements"].keys()))

    for page_num in page_numbers:
        # 각 페이지의 요약된 텍스트를 가져옴
        # 해당 페이지의 모든 이미지 요소에 대해 반복
        for image_element in state["section_elements"][page_num]["equation_elements"]:
            # 이미지 ID를 정수로 변환
            image_id = int(image_element["id"])

            # 데이터 배치에 이미지 정보, 관련 텍스트, 페이지 번호, ID를 추가
            data_batches.append(
                {
                    "image": state["equation"][image_id],  # 이미지 파일 경로
                    "page": page_num,  # 페이지 번호
                    "id": image_id,  # 이미지 ID
                }
            )
    # 생성된 데이터 배치를 GraphState 객체에 담아 반환
    return GraphState(equation_summary_data_batches=data_batches)

def create_image_summary_data_batches(state: GraphState):
    # 이미지 요약을 위한 데이터 배치를 생성하는 함수
    data_batches = []

    # 페이지 번호를 오름차순으로 정렬
    page_numbers = sorted(list(state["section_elements"].keys()))

    for page_num in page_numbers:
        # 각 페이지의 요약된 텍스트를 가져옴
        text = state["text_summary"][page_num]
        # 해당 페이지의 모든 이미지 요소에 대해 반복
        for image_element in state["section_elements"][page_num]["image_elements"]:
            # 이미지 ID를 정수로 변환
            image_id = int(image_element["id"])

            # 데이터 배치에 이미지 정보, 관련 텍스트, 페이지 번호, ID를 추가
            data_batches.append(
                {
                    "image": state["images"][image_id],  # 이미지 파일 경로
                    "text": text,  # 관련 텍스트 요약
                    "page": page_num,  # 페이지 번호
                    "id": image_id,  # 이미지 ID
                }
            )
    # 생성된 데이터 배치를 GraphState 객체에 담아 반환
    return GraphState(image_summary_data_batches=data_batches)

def create_table_summary_data_batches(state: GraphState):
    # 테이블 요약을 위한 데이터 배치를 생성하는 함수
    data_batches = []

    # 페이지 번호를 오름차순으로 정렬
    page_numbers = sorted(list(state["section_elements"].keys()))

    for page_num in page_numbers:
        # 각 페이지의 요약된 텍스트를 가져옴
        text = state["text_summary"][int(page_num)]
        # 해당 페이지의 모든 테이블 요소에 대해 반복
        for image_element in state["section_elements"][page_num]["table_elements"]:
            # 테이블 ID를 정수로 변환
            image_id = int(image_element["id"])

            # 데이터 배치에 테이블 정보, 관련 텍스트, 페이지 번호, ID를 추가
            data_batches.append(
                {
                    "table": state["tables"][image_id],  # 테이블 데이터
                    "text": text,  # 관련 텍스트 요약
                    "page": page_num,  # 페이지 번호
                    "id": image_id,  # 테이블 ID
                }
            )
    # 생성된 데이터 배치를 GraphState 객체에 담아 반환
    return GraphState(table_summary_data_batches=data_batches)

from langchain_teddynote.models import MultiModal
from langchain_core.runnables import chain


@chain
def extract_image_summary(data_batches):
    # 객체 생성
    llm = ChatOpenAI(
        temperature=0,  # 창의성 (0.0 ~ 2.0)
        model_name="gpt-4o-mini",  # 모델명
    )

    system_prompt = """You are an expert in extracting useful information from IMAGE.
    With a given image, your task is to extract key entities, summarize them, and write useful information that can be used later for retrieval."""

    image_paths = []
    system_prompts = []
    user_prompts = []

    for data_batch in data_batches:
        context = data_batch["text"]
        image_path = data_batch["image"]
        user_prompt_template = f"""Here is the context related to the image: {context}
        
###

Output Format:

<image>
<title>
<summary>
<entities> 
</image>

"""
        image_paths.append(image_path)
        system_prompts.append(system_prompt)
        user_prompts.append(user_prompt_template)

    # 멀티모달 객체 생성
    multimodal_llm = MultiModal(llm)

    # 이미지 파일로 부터 질의
    answer = multimodal_llm.batch(
        image_paths, system_prompts, user_prompts, display_image=False
    )
    return answer


@chain
def extract_table_summary(data_batches):
    # 객체 생성
    llm = ChatOpenAI(
        temperature=0,  # 창의성 (0.0 ~ 2.0)
        model_name="gpt-4o-mini",  # 모델명
    )

    system_prompt = "You are an expert in extracting useful information from TABLE. With a given image, your task is to extract key entities, summarize them, and write useful information that can be used later for retrieval."

    image_paths = []
    system_prompts = []
    user_prompts = []

    for data_batch in data_batches:
        context = data_batch["text"]
        image_path = data_batch["table"]
        user_prompt_template = f"""Here is the context related to the image of table: {context}
        
###

Output Format:

<table>
<title>
<table_summary>
<key_entities> 
<data_insights>
</table>

"""
        image_paths.append(image_path)
        system_prompts.append(system_prompt)
        user_prompts.append(user_prompt_template)

    # 멀티모달 객체 생성
    multimodal_llm = MultiModal(llm)

    # 이미지 파일로 부터 질의
    answer = multimodal_llm.batch(
        image_paths, system_prompts, user_prompts, display_image=False
    )
    return answer

@chain
def extract_equation_summary(data_batches):
    # 객체 생성
    llm = ChatOpenAI(
        temperature=0,  # 창의성 (0.0 ~ 2.0)
        model_name="gpt-4o-mini",  # 모델명
    )

    system_prompt = """You are an expert in extracting LaTeX equations from images. 
    
When extracting equations, please follow these specific instructions:
1. Use a single backslash (`\`) for LaTeX commands. Do not use double backslashes (`\\`) unless it is specifically required for commands like `\\` for newlines.
2. In cases like `\lambda`, always use a single backslash, i.e., `\lambda` instead of `\\lambda`.
3. Use the equals sign (`=`) directly in equations, and do not replace it with `\=`.
4. Ensure the extracted LaTeX equations are syntactically correct and avoid introducing unnecessary escape characters.

"""

    image_paths = []
    system_prompts = []
    user_prompts = []

    for data_batch in data_batches:
        image_path = data_batch["image"]
        user_prompt_template = f"""
        
###

Output Format:
$$ 
<equation>
$$

"""
        image_paths.append(image_path)
        system_prompts.append(system_prompt)
        user_prompts.append(user_prompt_template)

    # 멀티모달 객체 생성
    multimodal_llm = MultiModal(llm)

    # 이미지 파일로 부터 질의
    answer = multimodal_llm.batch(
        image_paths, system_prompts, user_prompts, display_image=False
    )
    return answer




def create_equation_summary(state: GraphState):
    # 이미지 요약 추출
    # extract_image_summary 함수를 호출하여 이미지 요약 생성
    equation_summaries = extract_equation_summary.invoke(
        state["equation_summary_data_batches"],
    )

    # 이미지 요약 결과를 저장할 딕셔너리 초기화
    equation_summary_output = dict()

    # 각 데이터 배치와 이미지 요약을 순회하며 처리
    for data_batch, equation_summary in zip(
        state["equation_summary_data_batches"], equation_summaries
    ):
        # 데이터 배치의 ID를 키로 사용하여 이미지 요약 저장
        equation_summary_output[data_batch["id"]] = equation_summary

    # 이미지 요약 결과를 포함한 새로운 GraphState 객체 반환
    return GraphState(equation_summary=equation_summary_output)



def create_image_summary(state: GraphState):
    # 이미지 요약 추출
    # extract_image_summary 함수를 호출하여 이미지 요약 생성
    image_summaries = extract_image_summary.invoke(
        state["image_summary_data_batches"],
    )

    # 이미지 요약 결과를 저장할 딕셔너리 초기화
    image_summary_output = dict()

    # 각 데이터 배치와 이미지 요약을 순회하며 처리
    for data_batch, image_summary in zip(
        state["image_summary_data_batches"], image_summaries
    ):
        # 데이터 배치의 ID를 키로 사용하여 이미지 요약 저장
        image_summary_output[data_batch["id"]] = image_summary

    # 이미지 요약 결과를 포함한 새로운 GraphState 객체 반환
    return GraphState(image_summary=image_summary_output)

def create_table_summary(state: GraphState):
    # 테이블 요약 추출
    table_summaries = extract_table_summary.invoke(
        state["table_summary_data_batches"],
    )

    # 테이블 요약 결과를 저장할 딕셔너리 초기화
    table_summary_output = dict()

    # 각 데이터 배치와 테이블 요약을 순회하며 처리
    for data_batch, table_summary in zip(
        state["table_summary_data_batches"], table_summaries
    ):
        # 데이터 배치의 ID를 키로 사용하여 테이블 요약 저장
        table_summary_output[data_batch["id"]] = table_summary

    # 테이블 요약 결과를 포함한 새로운 GraphState 객체 반환
    return GraphState(table_summary=table_summary_output)

@chain
def table_markdown_extractor(data_batches):
    # 객체 생성
    llm = ChatOpenAI(
        temperature=0,  # 창의성 (0.0 ~ 2.0)
        model_name="gpt-4o",  # 모델명
    )

    system_prompt = "You are an expert in converting image of the TABLE into markdown format. Be sure to include all the information in the table. DO NOT narrate, just answer in markdown format."

    image_paths = []
    system_prompts = []
    user_prompts = []

    for data_batch in data_batches:
        image_path = data_batch["table"]
        user_prompt_template = f"""DO NOT wrap your answer in ```markdown``` or any XML tags.
        
###

Output Format:

<table_markdown>

"""
        image_paths.append(image_path)
        system_prompts.append(system_prompt)
        user_prompts.append(user_prompt_template)

    # 멀티모달 객체 생성
    multimodal_llm = MultiModal(llm)

    # 이미지 파일로 부터 질의
    answer = multimodal_llm.batch(
        image_paths, system_prompts, user_prompts, display_image=False
    )
    return answer


def create_table_markdown(state: GraphState):
    # table_markdown_extractor를 사용하여 테이블 마크다운 생성
    # state["table_summary_data_batches"]에 저장된 테이블 데이터를 사용
    table_markdowns = table_markdown_extractor.invoke(
        state["table_summary_data_batches"],
    )

    # 결과를 저장할 딕셔너리 초기화
    table_markdown_output = dict()

    # 각 데이터 배치와 생성된 테이블 마크다운을 매칭하여 저장
    for data_batch, table_summary in zip(
        state["table_summary_data_batches"], table_markdowns
    ):
        # 데이터 배치의 id를 키로 사용하여 테이블 마크다운 저장
        table_markdown_output[data_batch["id"]] = table_summary

    # 새로운 GraphState 객체 반환, table_markdown 키에 결과 저장
    return GraphState(table_markdown=table_markdown_output)

########################################### 번역 



from langchain_core.output_parsers import StrOutputParser
def get_translator(model_name, prompt):
    # ChatOpenAI 모델의 또 다른 인스턴스를 생성합니다. (이전 인스턴스와 동일한 설정)
    
    trans_prompt = PromptTemplate.from_template(prompt)
    
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=0,
    )

    # 문서 요약을 위한 체인을 생성합니다.
    # 이 체인은 여러 문서를 입력받아 하나의 요약된 텍스트로 결합합니다.
    # text_summary_chain = create_stuff_documents_chain(llm, trans_prompt)
    
    text_summary_chain = trans_prompt | llm | StrOutputParser()
    
    
    return text_summary_chain
