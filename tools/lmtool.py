import os
from typing import Annotated, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

BASE_DIR = "C:/local_workspace/prj_ecellm/"
os.chdir(BASE_DIR)

with open('./api_key/API_KEY.txt', 'r') as api:
    os.environ["OPENAI_API_KEY"] = api.read()
with open('./api_key/TAVILY_API.txt', 'r') as api:
    os.environ["TAVILY_API_KEY"] = api.read()
    
class AddItemMethod(BaseModel):
    queries:list[str] = Field(..., description= "생성된 이커머스 검색 쿼리", min_length=20)

def season_item_llm(entity:str, model:str="gpt-4o-mini", temperature:float=0.1):
    model = ChatOpenAI(model=model, temperature=temperature)
    model = model.with_structured_output(AddItemMethod)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
            - Goal -
            당신은 이커머스 플랫폼용 검색 쿼리를 생성하는 전문가입니다. 
            당신의의 역할은 “{entity}”와 관련된 시맨틱한 정보를 담은 사용자 검색 쿼리를 만드는 것입니다.

            """),
            ("human", """
            '{entity}'와 관련된 상품에 대한 컴색 문장을 생성하시오.
            
            검색 문장은 다음 조건을 만족해야 합니다:
            1. 검색 문장은 한 줄로 작성한다.
            2. {entity}와 관련된 개념이 문장 속에 하나 이상 포함되어야 한다.
            3. {entity}를 최대한 직접적으로 언급하지 않는 문장을 생성하여야 합니다.
            4. 검색 문장은 간결하고 명확하게 단답형으로 끝나야 한다. 

            예시:
            Entity: "해변"
            - "여름철 해변가에서 입기 좋은 반팔 티셔츠"
            - "햇볕이 강한 날씨에 사용할 선크림"
            - "7월달에 입기 좋은 바지"

            Entity: {entity}
            """)
        ]
    )
    
    chain = prompt | model
    try:
        result = chain.invoke({"entity": entity})
        return result.queries
    except:
        return None
    
# 구성요소 찾아서 트라이. 좋은데 후속 조치가 어려움.
class ItemMethod(BaseModel):
    elements:list[str] = Field(..., description= "밀접하게 연관된 구성요소", min_length=20)

def elements_item_llm(entity:str, model:str="gpt-4o-mini", temperature:float=0.1):
    model = ChatOpenAI(model=model, temperature=temperature)
    model = model.with_structured_output(ItemMethod)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
            - Goal -
            당신은 주어진 {entity}와 매우 밀접하게 관련된 여러가지 구성요소를 찾아내는 역할을 수행합니다.
            주어진 지시사항에 맞추어 이를 수행하시오.

            """),
            ("human", """
            '{entity}'와 매우 밀접하게 관련된 구성요소를 찾아냅니다.
            이러한 구성요소는 밀접한 상품, 인물, 장소, 행동 등이 될 수 있습니다.
            출력된 Elements가 {entity}와 매우 밀접하게 관련되어있는지 면밀하게 검토 후 적절한 Elements만 반환하시오.
            이렇게 출력되는 Elements는 주어진 {entity}를 제외한 다른 entity와의 관계성은 현저하게 낮아야 합니다.

            예시:
            Entity: "여름"
            Output: ['에어컨', '선풍기', '물놀이', '수박', '선크림']

            Entity: {entity}
            """)
        ]
    )
    
    chain = prompt | model
    try:
        result = chain.invoke({"entity": entity})
        return result.elements
    except:
        return None

# 계절성이라는 카테고리를 주고 무작위로 생성하기 만듦.
# 이후 해당 문장으로부터 적절한 매핑을 진행하는 방식.
class SeasonItemMethod(BaseModel):
    queries:list[str] = Field(..., description= "생성된 이커머스 검색 쿼리", min_length=20)

def season_gen_llm(Context:str, model:str="gpt-4o-mini", temperature:float=0.1):
    model = ChatOpenAI(model=model, temperature=temperature)
    model = model.with_structured_output(SeasonItemMethod)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
            - Goal -
            당신은 이커머스 플랫폼용 검색 쿼리를 생성하는 전문가입니다. 
            당신의의 역할은 계절성과 관련된 요소를 담은 검색용 쿼리를 생성하는 것입니다.

            """),
            ("human", """
            계절성과 관련된 요소를 담은 검색용 쿼리를 생성하시오.
            (Context)를 반영하여 적절한 문장을 생성하시오.
            
            검색 문장은 다음 조건을 만족해야 합니다:
            1. 검색 문장은 한 줄로 작성한다.
            2. 계절(봄, 여름, 가을, 겨울)을 직접적으로 언급하는 최대한 피한다.
            3. 검색 문장은 간결하고 명확하게 단답형으로 끝나야 한다. 

            예시:
            Context: "계절 활동과 관련"
            - "스키장에서 입기 좋은 패딩"
            - "햇볕이 강한 바닷가에 필요한 선크림"
            - "꽃놀이에 입을 시원한 카디건"
            - "단풍 구경에 입을 가벼운 바람막이"

            Context: {Context}
            """)
        ]
    )
    
    chain = prompt | model
    try:
        result = chain.invoke({"Context": Context})
        return result.queries
    except:
        return None

class SeasonCheckMethod(BaseModel):
    season:str = Field(..., description= "가장 연관성이 짙은 계절", enum=['봄', '여름', '가을', '겨울', '복합적'])
    relationship: str = Field(..., description= "계절과의 연관성 정도", enum=['매우강함', '강함', '보통', '약함', '매우약함'])
    keyword: str = Field(..., description= "주어진 문장 속 선택한 계절과 강한 연관성을 보이는 핵심 키워드")
    
def season_check_llm(text:str, model:str="gpt-4o-mini", temperature:float=0.1):
    model = ChatOpenAI(model=model, temperature=temperature)
    model = model.with_structured_output(SeasonCheckMethod)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
            - Goal -
            당신은 주어진 문장에서 발견되는 "계절" 요소를 찾아내는 역할을 수행합니다.
            주어진 문장을 보고 해당 문장이 어떤 계절과 가장 연관성이 깊은지 찾고, 그 연관성의 정도를 표현하시오.
            이후 주어진 문장에서 어떠한 키워드가 이러한 판단의 근거가 되었는지 찾아 제시하시오.

            """),
            ("human", """
            주어진 문장이 어떠한 계절과 가장 관계가 깊은지 찾아내시오.
            이 때 해당 계절과의 관계 정도를 ['매우강함', '강함', '보통', '약함', '매우약함']로 표현하시오.
            위와 같이 판단한 근거가 되는 핵심 키워드를 제시된 문장 속에서 찾아 제시하시오.
            
            예시:
            Sentence: "꽃 피는 시기에 어울리는 린넨 블라우스"
            Answer:
                season: "봄",
                relationship: "매우강함",
                keyword: "꽃 피는 시기"

            Sentence: {text}
            """)
        ]
    )
    
    chain = prompt | model
    try:
        result = chain.invoke({"text": text})
        return result
    except:
        return None
    
# 마찬가지로 성별에서 진행.
class GenderItemMethod(BaseModel):
    queries:list[str] = Field(..., description= "생성된 이커머스 검색 쿼리", min_length=20)

def gender_gen_llm(Context:str, model:str="gpt-4o-mini", temperature:float=0.1):
    model = ChatOpenAI(model=model, temperature=temperature)
    model = model.with_structured_output(GenderItemMethod)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
            - Goal -
            당신은 이커머스 플랫폼용 검색 쿼리를 생성하는 전문가입니다. 
            당신의의 역할은 성별과 밀접하게 관련된 요소를 담은 검색용 쿼리를 생성하는 것입니다.

            """),
            ("human", """
            성별(Gender)과 밀접하게 관련된 요소를 담은 검색용 쿼리를 생성하시오.
            (Context)를 반영하여 적절한 문장을 생성하시오.
            
            검색 문장은 다음 조건을 만족해야 합니다:
            1. 검색 문장은 한 줄로 작성한다.
            2. 성별(남성, 남자, 여성, 여자)을 직접적으로 언급하는 최대한 피한다.
            3. 검색 문장은 간결하고 명확하게 단답형으로 끝나야 한다. 

            예시:
            Context: "성별과 연관된 인물과 관련"
            - "엄마가 쓰기 좋은 우산"
            - "아빠가 쓰기 좋은 컴퓨터"
            - "임산부에게 필요한 의상"
            - "장모님께 드릴 선물"

            Context: {Context}
            """)
        ]
    )
    
    chain = prompt | model
    try:
        result = chain.invoke({"Context": Context})
        return result.queries
    except:
        return None

class GenderCheckMethod(BaseModel):
    gender:str = Field(..., description= "가장 연관성이 짙은 성별", enum=['남성', '여성', '복합적'])
    relationship: str = Field(..., description= "성별과의 연관성 정도", enum=['매우강함', '강함', '보통', '약함', '매우약함'])
    keyword: str = Field(..., description= "주어진 문장 속 선택한 성별과 강한 연관성을 보이는 핵심 키워드")
    
def gender_check_llm(text:str, model:str="gpt-4o-mini", temperature:float=0.1):
    model = ChatOpenAI(model=model, temperature=temperature)
    model = model.with_structured_output(GenderCheckMethod)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
            - Goal -
            당신은 주어진 문장에서 발견되는 "성별" 요소를 찾아내는 역할을 수행합니다.
            주어진 문장을 보고 해당 문장이 어떤 성별과 가장 연관성이 깊은지 찾고, 그 연관성의 정도를 표현하시오.
            이후 주어진 문장에서 어떠한 키워드가 이러한 판단의 근거가 되었는지 찾아 제시하시오.

            """),
            ("human", """
            주어진 문장이 어떠한 성별과 가장 관계가 깊은지 찾아내시오.
            이 때 해당 성별과의 관계 정도를 ['매우강함', '강함', '보통', '약함', '매우약함']로 표현하시오.
            위와 같이 판단한 근거가 되는 핵심 키워드를 제시된 문장 속에서 찾아 제시하시오.
            
            예시:
            Sentence: "장모님을 위한 선물용 스카프프"
            Answer:
                gender: "여성",
                relationship: "매우강함",
                keyword: "장모님"

            Sentence: {text}
            """)
        ]
    )
    
    chain = prompt | model
    try:
        result = chain.invoke({"text": text})
        return result
    except:
        return None


# 마찬가지로 색상에서 진행.
class ColorItemMethod(BaseModel):
    queries:list[str] = Field(..., description= "생성된 이커머스 검색 쿼리", min_length=20)

def color_gen_llm(Context:str, model:str="gpt-4o-mini", temperature:float=0.1):
    model = ChatOpenAI(model=model, temperature=temperature)
    model = model.with_structured_output(ColorItemMethod)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
            - Goal -
            당신은 이커머스 플랫폼용 검색 쿼리를 생성하는 전문가입니다. 
            당신의의 역할은 색상과 밀접하게 관련된 요소를 담은 검색용 쿼리를 생성하는 것입니다.

            """),
            ("human", """
            색상(Color)과 밀접하게 관련된 요소를 담은 검색용 쿼리를 생성하시오.
            (Context)를 반영하여 적절한 문장을 생성하시오.
            
            검색 문장은 다음 조건을 만족해야 합니다:
            1. 검색 문장은 한 줄로 작성한다.
            2. 특정한 색에 대한 단서가 문장 속 반드시 포함되어야 한다.
            3. 검색 문장은 간결하고 명확하게 단답형으로 끝나야 한다. 

            예시:
            Context: "색상과 연관된 의류와 관련"
            - "검정 색상 가죽 재킷"
            - "민트색 반팔 티셔츠"
            - "노란색 포인트가 있는 가죽 구두"
            - "빨간색으로 포인트를 준 겨울용 패딩"

            Context: {Context}
            """)
        ]
    )
    
    chain = prompt | model
    try:
        result = chain.invoke({"Context": Context})
        return result.queries
    except:
        return None

class ColorCheckMethod(BaseModel):
    best_fit_color:str = Field(..., description= "가장 연관성이 짙은 색상", enum=['검정색', '남색', '회색', '은색', '빨간색', '주황색', '노랑색', '초록색', '파랑색', '보라색', '분홍', '하얀색', '갈색', '금색', '베이지색', '민트색', '청록색', '하늘색', '라임색', '혼합색상', '투명색', '버건디', '차콜색', '카키색', '없음'])
    second_fit_color:Optional[str] = Field(..., description= "두 번째로 연관성이 짙은 색상", enum=['검정색', '남색', '회색', '은색', '빨간색', '주황색', '노랑색', '초록색', '파랑색', '보라색', '분홍', '하얀색', '갈색', '금색', '베이지색', '민트색', '청록색', '하늘색', '라임색', '혼합색상', '투명색', '버건디', '차콜색', '카키색','없음'])
    third_fit_color:Optional[str] = Field(..., description= "세 번째로 연관성이 짙은 색상", enum=['검정색', '남색', '회색', '은색', '빨간색', '주황색', '노랑색', '초록색', '파랑색', '보라색', '분홍', '하얀색', '갈색', '금색', '베이지색', '민트색', '청록색', '하늘색', '라임색', '혼합색상', '투명색', '버건디', '차콜색', '카키색', '없음'])
    relationship: str = Field(..., description= "색상과의 연관성 정도", enum=['매우강함', '강함', '보통', '약함', '매우약함'])
    keyword: str = Field(..., description= "주어진 문장 속 선택한 색상과 강한 연관성을 보이는 핵심 키워드")
    
def color_check_llm(text:str, model:str="gpt-4o-mini", temperature:float=0.1):
    model = ChatOpenAI(model=model, temperature=temperature)
    model = model.with_structured_output(ColorCheckMethod)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
            - Goal -
            당신은 주어진 문장에서 발견되는 "색상" 요소를 찾아내는 역할을 수행합니다.
            주어진 문장을 보고 해당 문장이 어떤 색상과 가장 연관성이 깊은지 찾고, 그 연관성의 정도를 표현하시오.
            이후 주어진 문장에서 어떠한 키워드가 이러한 판단의 근거가 되었는지 찾아 제시하시오.

            """),
            ("human", """
            주어진 문장이 어떠한 색상과 가장 관계가 깊은지 찾아내시오.
            이 때 해당 색상과의 관계 정도를 ['매우강함', '강함', '보통', '약함', '매우약함']로 표현하시오.
            위와 같이 판단한 근거가 되는 핵심 키워드를 제시된 문장 속에서 찾아 제시하시오.
            
            예시:
            Sentence: "노란색 포인트가 있는 가죽 구두"
            Answer:
                best_fit_color: "노랑색",
                second_fit_color: "라임색",
                third_fit_color: "없음",
                relationship: "매우강함",
                keyword: "노란색"

            Sentence: {text}
            """)
        ]
    )
    
    chain = prompt | model
    try:
        result = chain.invoke({"text": text})
        return result
    except:
        return None