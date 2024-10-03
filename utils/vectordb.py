
import os 
import pickle
from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from utils.config import config

def get_embedding(config):
    # 임베딩 모델 설정
    # model_path = config['embed_model']
    # model_kwargs = {'device': 'cuda'}
    # encode_kwargs = {'normalize_embeddings': True}
    # embeddings = HuggingFaceEmbeddings(
    #     model_name=model_path,
    #     model_kwargs=model_kwargs,
    #     encode_kwargs=encode_kwargs
    # )
    # # if config['embed_model']=='gpt'
    embeddings = OpenAIEmbeddings(model=config['embed_model'])
    # embeddings = OpenAIEmbeddings()
    return embeddings

def build_db(data):
    
    text_splitter = RecursiveCharacterTextSplitter(
    # 청크 크기를 매우 작게 설정합니다. 예시를 위한 설정입니다.
    chunk_size=1000,
    # 청크 간의 중복되는 문자 수를 설정합니다.
    chunk_overlap=50,
    # 문자열 길이를 계산하는 함수를 지정합니다.
    length_function=len,
    # 구분자로 정규식을 사용할지 여부를 설정합니다.
    is_separator_regex=False,
    )
    
    docs = []
    for key, value in data['section_elements'].items():
        section_name = data['section_names'][int(key)]
        
        if value['image_elements']:
            for element in value['image_elements']:
                category = element['category']
                page = element['page']
                id = element['id']
                try:
                    print(list(data['images_summary'].keys()))
                    image_summary = data['images_summary'][str(id)]
                except:
                    print(list(data['images_summary'].keys()))
                    image_summary = data['images_summary'][str(id)]
                
                try:
                    print(list(data['images'].keys()))
                    image_path = data['images'][str(id)]
                except:
                    print(list(data['images'].keys()))
                    image_path = data['images'][str(id)]
                    
                img_text = ''
                for img_ele in data['image_summary_data_batches']:
                    if img_ele['id'] == str(id):
                        img_text = img_ele['text']
                        break
                
                metadata = {
                    'category': category,
                    'section' : section_name,
                    'page': page,
                    'id': id, 
                    'image_path': image_path,
                    'image_summary': image_summary
                }
                page_content = img_text
                
                doc = Document(page_content=page_content, metadata=metadata)
                docs.append(doc)
                
        if value['table_elements']:
            for element in value['table_elements']:
                category = element['category']
                page = element['page']
                id = element['id']
                
                try:
                    print(list(data['tables_summary'].keys()))
                    table_summary = data['tables_summary'][str(id)]
                except:
                    print(list(data['tables_summary'].keys()))
                    table_summary = data['tables_summary'][str(id)]
                
                try:
                    print(list(data['tables'].keys()))
                    table_path = data['tables'][str(id)]
                except:
                    print(list(data['tables'].keys()))
                    table_path = data['tables'][str(id)]
                
                table_text = ''
                for tab_ele in data['table_summary_data_batches']:
                    if tab_ele['id'] == str(id):
                        table_text = tab_ele['text']
                        break
                
                metadata = {
                    'category': category,
                    'section' : section_name,
                    'page': page,
                    'id': id, 
                    'table_path': table_path,
                    'table_summary': table_summary
                }
                page_content = table_text
                
                doc = Document(page_content=page_content, metadata=metadata)
                docs.append(doc)
                
        for element in value['text_elements']:
            category = element['category']
            page = element['page']
            id = element['id']
            section_summary = data['texts_summary'][str(key)] # data['paper_summary']
            metadata = {
                    'category': category,
                    'section' : section_name,
                    'page': page,
                    'id': id,
                    'section_summary': section_summary
                }
            
            page_content = element['text']
            
            doc = Document(page_content=page_content, metadata=metadata)
            docs.append(doc)

    split_documents = text_splitter.split_documents(docs)
    save_path = os.path.splitext(data['filepath'])[0]
    save_db_file = f'{save_path}/db'
    with open(f'{save_db_file}.pkl', 'wb') as f:
        pickle.dump(split_documents, f)
        
    # Chroma DB 생성 및 반환 persist_directory=DB_PATH
    db = FAISS.from_documents(split_documents, embedding=get_embedding(config))
    db.save_local(save_db_file)
    
    print('build db done')