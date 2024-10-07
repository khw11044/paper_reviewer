config = {
    "llm_predictor": {
        "model_name": "llama3.1-dacon-Q8:latest",   # llama3.1-dacon-Q8:latest,  llama3.1
        "temperature": 0
    },
    # "intfloat/multilingual-e5-small", intfloat/multilingual-e5-base, intfloat/multilingual-e5-large
    # BAAI/bge-m3
    "embed_model": "text-embedding-3-small", # "intfloat/multilingual-e5-large",  
    
    
    "search_type": "mmr",           # "mmr"  similarity   similarity_score_threshold
    'search_kwargs_k': 3,           # 유사도 기반 상위 k개
    'search_kwargs_lambda': 0.5,    # 0~1, 0에 가까울수록 다양성, 1에 가까울수록 관련성
    'score_threshold': 0.4,         # 0~1, 쿼리 문장과 최소한 0.x 이상의 유사도를 가진 문서만
    
    'bm25_k' : 3,                   # 검색어 기반 상위 k개
    "ensemble_search_type": "mmr",  # 앙상블 서칭 타입
    "ensemble_weight": [0.3,0.7],   # bm25와 vector 가중치


}