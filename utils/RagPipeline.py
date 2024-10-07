# langchain_community.llms import Ollama
# from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain.prompts import PromptTemplate

from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
# from langchain_community.vectorstores import Chroma
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever
# from langchain_teddynote.retrievers import KiwiBM25Retriever

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import HumanMessage



import pickle

from utils.prompt import contextualize_q_prompt, qa_prompt
from utils.prompt import template
from utils.vectordb import get_embedding

def format_docs(docs):
    """검색된 문서들을 하나의 문자열로 포맷팅"""

    context = ""
    
    for doc in docs:
        metadata = doc.metadata
        section = metadata['section']
        if metadata['category'] == 'figure':
            summary = metadata['image_summary']
        else:
            summary = metadata['section_summary']
        metadata_context = f"#Section : {section} \n #Summary: {summary} \n #Content: {doc.page_content}"
        context += metadata_context
        context += '\n\n'
        
    return context

class Ragpipeline:
    def __init__(self, source, config):
        # 1. LLM 바꿔보면서 실험하기 
        # self.llm = ChatOllama(model=config['llm_predictor']['model_name'], temperature=config['llm_predictor']['temperature'])
        # self.llm = ChatOllama(model="llama3-ko-instruct", temperature=0)
        self.llm = ChatOpenAI(model_name=config['llm_predictor']['model_name'], temperature=0.1)
        self.source = source
        self.config = config
        self.base_retriever = self.init_retriever()
        self.ensemble_retriever = self.init_ensemble_retriever()
        self.chain = self.init_chain()
        
        
    def init_retriever(self):
        source = self.source
        config = self.config
        
        # vector_store = Chroma(persist_directory=source, embedding_function=get_embedding())
        embeddings_model = get_embedding(config)  # config
        vector_store = FAISS.load_local(source, embeddings_model, allow_dangerous_deserialization=True)
        
        if config['search_type']=='mmr':
            retriever = vector_store.as_retriever(
                    search_type=config['search_type'],                                              # mmr 검색 방법으로 
                    # search_kwargs={'fetch_k': 5, "k": 3, 'lambda_mult': 0.4},      # 상위 10개의 관련 context에서 최종 5개를 추리고 'lambda_mult'는 관련성과 다양성 사이의 균형을 조정하는 파라메타 default 값이 0.5
                    search_kwargs={'fetch_k': config['search_kwargs_k']*2, "k": config['search_kwargs_k'], 'lambda_mult': config['search_kwargs_lambda']}, 
                )
        elif config['search_type']=='similarity':
            retriever = vector_store.as_retriever(
                    search_type=config['search_type'],                                              # mmr 검색 방법으로 
                    search_kwargs={"k": config['search_kwargs_k']}, 
                )
        else:
            retriever = vector_store.as_retriever(
                    search_type=self.config['search_type'],                                              # mmr 검색 방법으로 
                    search_kwargs={"k": config['search_kwargs_k'], 'score_threshold': config['score_threshold']}, 
                )
        
        return retriever
    
    def init_ensemble_retriever(self):
        source = self.source
        config = self.config
        
        retriever = self.base_retriever
        
        all_docs = pickle.load(open(f'{source}.pkl', 'rb'))
        bm25_retriever = BM25Retriever.from_documents(all_docs) # KiwiBM25Retriever.from_documents(all_docs)
        bm25_retriever.k = config['bm25_k']
        
        ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, retriever],
                weights=config['ensemble_weight'],
                search_type=config['ensemble_search_type']
            )
        
        return ensemble_retriever
        
    def init_chain(self):
        retriever = self.ensemble_retriever       # get_retriever()
        
        # 1. 이어지는 대화가 되도록 대화기록과 체인
        history_aware_retriever = create_history_aware_retriever(self.llm, retriever, contextualize_q_prompt)      # self.mq_ensemble_retriever
        
        # 2. 문서들의 내용을 답변할 수 있도록 리트리버와 체인
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        
        # 3. 1과 2를 합침
        chat_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | chat_chain
            | StrOutputParser()
        )
        
        return rag_chain
        
    
    def answer_generation(self, question: str) -> dict:
        full_response = self.chain.invoke(question)
        return full_response