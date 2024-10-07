from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder



template = """You are the author of the referenced paper and possess an in-depth understanding of its content, more than anyone else. 
You are fluent in both English and Korean. When answering the question, respond in Korean, but keep key terms, keywords, and technical terminology in English. 
Make sure to include the source of your answer, referencing the specific section or page number of the paper.


#Context: 
{context}

#Question:
{question}

#Answer:
"""


# 1. 사용자 질문 맥락화 프롬프트
contextualize_q_system_prompt = """
주요 목표는 사용자의 질문을 이해하기 쉽게 다시 작성하는 것입니다.
사용자의 질문과 채팅 기록이 주어졌을 때, 채팅 기록의 맥락을 참조할 수 있습니다.
채팅 기록이 없더라도 이해할 수 있는 독립적인 질문으로 작성하세요.
질문에 바로 대답하지 말고, 필요하다면 질문을 다시 작성하세요. 그렇지 않다면 질문을 그대로 반환합니다.        
"""
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


# 2. 질문 프롬프트
qa_system_prompt = """
You are the author of the referenced paper and possess an in-depth understanding of its content, more than anyone else. 
You are fluent in both English and Korean. When answering the question, respond in Korean, but keep key terms, keywords, and technical terminology in English. 
Make sure to include the source of your answer, referencing the specific section or page number of the paper.


#Context: 
{context}

#Question:
{question}

#Answer:
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

# 요약을 위한 프롬프트 템플릿을 정의합니다.
summary_prompt = """Please summarize the sentence according to the following REQUEST.
    
REQUEST:
1. Summarize the main points in bullet points.
3. Write the summary in same language as the context.
4. DO NOT translate any technical terms.
5. DO NOT include any unnecessary information.
6. Summary must include important entities, numerical values.

CONTEXT:
{context}

SUMMARY:"
"""

# 논문 전체 요약을 위한 map-reduce 프롬프트 템플릿을 정의합니다.
map_prompt = """You are a professional summarizer. 
You are given a summary list of documents and you make a summary list of this within 1 to 10 lines.
Please create a single summary of the documents according to the following REQUEST.
    
REQUEST:
1. Extract main points from a list of summaries of documents
2. Make final summaries in bullet points format.
2. Each summarized sentence must start with an emoji that fits the meaning of the each sentence.
3. Use various emojis to make the summary more interesting.
4. Write the summary in same language as the context.
5. DO NOT translate any technical terms.
6. DO NOT include any unnecessary information.


Here is a list of summaries of documents: 
{context}

SUMMARY:"
"""


trans_prompt = """You are a translator specializing in academic papers.
    Your task is to translate an English paper into Korean.
    Please follow the given instructions carefully.

    REQUEST:
    1. Translate the content into Korean.
    2. Do not translate technical terms or key concepts; keep them in English (e.g., Cross Attention, Transformer).
    3. Maintain the original meaning without adding any unnecessary information.
    4. Make sure to preserve important entities and numerical values.
    5. Use Korean translations for natural expressions but leave awkward terms in English if necessary.

    CONTEXT:
    {context}

    TRANSLATION:"""
