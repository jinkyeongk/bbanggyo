import os
from pymongo import MongoClient
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("❌ OpenAI API 키를 찾을 수 없습니다.")

# MongoDB 연결 (연결 문자열과 DB/컬렉션 이름을 실제 환경에 맞게 수정)
client = MongoClient("mongodb://localhost:27017")
db = client["crawlled"]         # 예: 'bakery_db'
collection = db["bakeries"]       # 예: 'bakeries'

# =====================================
# MongoDB에서 빵집 데이터 읽어오기 및 Document 생성
# =====================================
bakery_data = list(collection.find())

documents = []
for data in bakery_data:
    title = data.get("title", "제목 없음")
    scores = data.get("scores", {})
    total_score = scores.get("total_score", "N/A")
    taste_score = scores.get("taste_score", "N/A")
    price_score = scores.get("price_score", "N/A")
    cs_score = scores.get("cs_score", "N/A")
    
    # 각 리뷰 정보를 결합 (여러 리뷰가 있을 경우 줄바꿈으로 구분)
    review_texts = []
    for review in data.get("reviews", []):
        content = review.get("content", "")
        score = review.get("score", "")
        keywords = ", ".join(review.get("keywords", []))
        review_texts.append(f"리뷰: {content} (평점: {score}, 키워드: {keywords})")
    reviews_combined = "\n".join(review_texts)
    
    # 제목, 평점, 리뷰를 하나의 텍스트로 결합
    content_text = (
        f"빵집 이름: {title}\n"
        f"평점: 총점 {total_score}, 맛 {taste_score}, 가격 {price_score}, 고객서비스 {cs_score}\n"
        f"{reviews_combined}"
    )
    
    # LangChain Document 생성 (메타데이터로 _id나 제목을 추가할 수 있음)
    documents.append(Document(page_content=content_text, metadata={"_id": str(data.get("_id")), "title": title}))

# =====================================
# Chroma 벡터스토어 생성 (임베딩 생성)
# =====================================
embedding_function = OpenAIEmbeddings()
# collection_name은 벡터스토어 내의 하위 컬렉션 이름입니다.
chroma_store = Chroma.from_documents(documents, embedding_function, collection_name="bakery_vector_store")

# =====================================
# 메인 코드: 사용자 성격 기반 빵집 추천
# =====================================

# 1. 테스트용 하드코딩된 사용자 성격 설명
personality_query = (
    "나는 창의적이고 모던한 분위기를 좋아하며, 유쾌하고 독특한 경험을 추구하는 성격이야. "
    "새로운 맛과 재미있는 이야기가 있는 빵집을 원해."
)

# 2. Chroma 벡터스토어에서 사용자 성격과 유사한 빵집 문서를 검색 (예: 상위 3개)
similar_docs = chroma_store.similarity_search(personality_query, k=3)

# 3. 추천 프롬프트 구성  
#    - 후보 빵집들의 정보를 나열하고, 재미있는 추천과 추천 이유를 요청하는 형태로 구성
recommendation_prompt = f"사용자의 성격: {personality_query}\n\n"
recommendation_prompt += "다음 빵집 후보들 중에서 사용자에게 가장 어울리는 빵집을 하나 재미있게 추천해줘:\n"
for doc in similar_docs:
    recommendation_prompt += f"- {doc.page_content}\n"
# recommendation_prompt += "\n재미있는 추천과 함께, 왜 이 빵집을 추천했는지 상세하게 설명해줘."

# 4. ChatGPT API 호출 (추천 답변 생성)
llm = ChatOpenAI(temperature=0.7)
recommendation = llm([HumanMessage(content=recommendation_prompt)])

print("추천 결과:")
print(recommendation.content)

# 5. 추천 이유만 별도로 물어보기  
#    (추천 결과의 빵집 정보를 기반으로 자세한 설명을 추가 요청)
explanation_prompt = (
    f"위의 추천 결과에 대해, 왜 해당 빵집을 추천했는지 구체적인 이유를 다시 한 번 설명해줘.\n\n"
    f"사용자 성격: {personality_query}\n\n"
    f"추천된 빵집 정보: {similar_docs[0].page_content}"
)

explanation = llm([HumanMessage(content=explanation_prompt)])

print("\n추천 이유:")
print(explanation.content)
