import json
import os
import re
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import logging
from rank_bm25 import BM25Okapi
import json

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("jeju_dialect_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("JejuDialectSystem")

class JejuDialectProcessor:
    """
    제주어 방언 데이터 처리 클래스
    """
    def __init__(self, data_path=None):
        self.data = None
        self.utterances = []
        self.speakers = []
        self.dialect_mappings = []
        self.dialect_dict = {}
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path):
        """
        제주어 방언 데이터를 로드합니다.
        
        Args:
            data_path (str): JSON 파일 경로
        
        Returns:
            bool: 로드 성공 여부
        """
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            
            # 발화, 화자, 방언 매핑 추출
            self._extract_utterances()
            self._extract_speakers()
            self._extract_dialect_mappings()
            self._create_dialect_dictionary()
            
            logger.info(f"데이터 로드 성공: {data_path}")
            logger.info(f"발화 수: {len(self.utterances)}")
            logger.info(f"화자 수: {len(self.speakers)}")
            logger.info(f"방언 매핑 수: {len(self.dialect_mappings)}")
            logger.info(f"방언 사전 크기: {len(self.dialect_dict)}")
            
            return True
        except Exception as e:
            logger.error(f"데이터 로드 실패: {e}")
            return False
    
    def _extract_utterances(self):
        """발화 데이터를 추출합니다."""
        if self.data and 'utterance' in self.data:
            self.utterances = self.data['utterance']
    
    def _extract_speakers(self):
        """화자 정보를 추출합니다."""
        if self.data and 'speaker' in self.data:
            self.speakers = self.data['speaker']
    
    def _extract_dialect_mappings(self):
        """제주어와 표준어 매핑을 추출합니다."""
        self.dialect_mappings = []
        
        for utterance in self.utterances:
            if 'eojeolList' in utterance:
                eojeols = utterance['eojeolList']
                for eojeol in eojeols:
                    if eojeol.get('isDialect', False):
                        dialect = eojeol.get('eojeol', '')
                        standard = eojeol.get('standard', '')
                        if dialect and standard:
                            self.dialect_mappings.append((dialect, standard))
    
    def _create_dialect_dictionary(self):
        """제주어-표준어 사전을 생성합니다."""
        self.dialect_dict = {}
        for dialect, standard in self.dialect_mappings:
            if dialect not in self.dialect_dict:
                self.dialect_dict[dialect] = standard
    
    def get_pattern_analysis(self):
        """
        제주어 패턴을 분석합니다.
        
        Returns:
            dict: 분석 결과
        """
        # 제주어 접미사/접두사 패턴 분석
        suffixes = []
        prefixes = []
        char_replacements = []
        
        for dialect, standard in self.dialect_mappings:
            # 접미사 분석 (방언이 표준어보다 길 경우)
            if len(dialect) > len(standard) and dialect.startswith(standard[:2]):
                suffix = dialect[len(standard):]
                if suffix:
                    suffixes.append(suffix)
            
            # 접두사 분석
            if len(dialect) > len(standard) and dialect.endswith(standard[-2:]):
                prefix = dialect[:len(dialect)-len(standard)]
                if prefix:
                    prefixes.append(prefix)
            
            # 문자 변형 패턴 분석
            if len(dialect) == len(standard):
                for i in range(len(dialect)):
                    if i < len(standard) and dialect[i] != standard[i]:
                        char_replacements.append((standard[i], dialect[i]))
        
        # 빈도 분석
        suffix_counter = Counter(suffixes)
        prefix_counter = Counter(prefixes)
        replacement_counter = Counter(char_replacements)
        
        return {
            'most_common_suffixes': suffix_counter.most_common(10),
            'most_common_prefixes': prefix_counter.most_common(10),
            'most_common_replacements': replacement_counter.most_common(10)
        }

    def extract_key_components(self, text):
        """
        텍스트에서 핵심 구성요소를 추출합니다.
        
        Args:
            text (str): 분석할 텍스트
        
        Returns:
            dict: 고수준, 저수준, 구조적 구성요소
        """
        # 1. 고수준 구성요소 (의미적 개념, 목적)
        high_level = " ".join(text.split()[:min(15, len(text.split()))])
        
        # 2. 저수준 구성요소 (세부 정보, 방언 단어)
        dialect_words = []
        for word in text.split():
            if word in self.dialect_dict:
                dialect_words.append(word)
        low_level = " ".join(dialect_words) if dialect_words else text
        
        # 3. 구조적 구성요소 (문장 구조, 패턴)
        words = text.split()
        structural = " ".join([words[0]] + [words[-1]] if len(words) > 1 else words)
        
        return {
            "high_level": high_level,
            "low_level": low_level,
            "structural": structural
        }

class JejuDialectCoT:
    """
    제주어 방언 Chain-of-Thought(CoT) 추론 시스템
    HDLCoRe 논문의 HDL-aware CoT 개념을 제주어 도메인에 적용
    """
    def __init__(self, processor, llm_model_name="anthropic/voyage"):
        self.processor = processor
        self.model_name = llm_model_name
        
        # Anthropic API 키 설정
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY 환경 변수가 설정되지 않았습니다.")
            self.anthropic_client = None
            return
            
        try:
            from anthropic import Anthropic
            self.anthropic_client = Anthropic(api_key=self.api_key)
            logger.info(f"Voyage 모델 로드 성공: {llm_model_name}")
        except Exception as e:
            logger.error(f"Voyage 모델 로드 실패: {e}")
            self.anthropic_client = None

    def _classify_query(self, query):
        """
        쿼리를 분류합니다. (HDLCoRe 논문의 작업 유형 및 복잡성 분류 개념 적용)
        
        Args:
            query (str): 분석할 쿼리
        
        Returns:
            tuple: (복잡성, 유형) - ('simple'/'complex', 'translation'/'analysis'/'question')
        """
        # 방언 단어 비율로 복잡성 판단
        dialect_words = 0
        total_words = len(query.split())
        
        for word in query.split():
            if word in self.processor.dialect_dict:
                dialect_words += 1
        
        # 방언 비율이 30% 이상이면 복잡한 쿼리로 간주
        complexity = "complex" if dialect_words / total_words >= 0.3 else "simple"
        
        # 쿼리 유형 분류
        if "번역" in query or "뜻" in query:
            query_type = "translation"
        elif "분석" in query or "패턴" in query or "특징" in query:
            query_type = "analysis"
        else:
            query_type = "question"
        
        return complexity, query_type
    
    def generate_cot_prompt(self, query, retrieval_response=None):
        """
        쿼리에 대한 Chain-of-Thought 프롬프트를 생성합니다.
        
        Args:
            query (str): 사용자 쿼리
            retrieval_response (dict): Contextual Retrieval 응답
        
        Returns:
            str: CoT 프롬프트
        """
        # 쿼리 분류
        complexity, query_type = self._classify_query(query)
        
        # 기본 프롬프트 템플릿
        prompt = f"제주어 전문가로서 다음 내용을 처리해주세요: \"{query}\"\n\n"
        
        # 복잡성 및 유형에 따른 프롬프트 추가
        if complexity == "simple":
            if query_type == "translation":
                prompt += "단계적으로 생각하며 제주어를 표준어로 번역하겠습니다.\n"
                prompt += "1. 제주어 단어 식별하기\n"
                prompt += "2. 각 제주어 단어의 표준어 의미 찾기\n"
                prompt += "3. 문맥에 맞게 번역하기\n\n"
            elif query_type == "analysis":
                prompt += "단계적으로 생각하며 제주어 특성을 분석하겠습니다.\n"
                prompt += "1. 주요 제주어 표현 식별하기\n"
                prompt += "2. 제주어의 언어적 특성 분석하기\n"
                prompt += "3. 표준어와의 차이점 설명하기\n\n"
            else:
                prompt += "단계적으로 생각하며 질문에 답변하겠습니다.\n"
                prompt += "1. 제주어 관련 핵심 정보 식별하기\n"
                prompt += "2. 질문의 의도 파악하기\n"
                prompt += "3. 관련 정보를 활용하여 답변 구성하기\n\n"
        else:  # complex
            if query_type == "translation":
                prompt += "복잡한 제주어 표현이 포함된 내용을 단계적으로 번역하겠습니다.\n"
                prompt += "1. 제주어 단어와 표현 식별하기\n"
                prompt += "2. 문법적 특성 분석하기\n"
                prompt += "3. 제주어 표현의 문화적 맥락 고려하기\n"
                prompt += "4. 표준어로 정확하게 번역하기\n"
                prompt += "5. 번역 결과 검증하기\n\n"
            elif query_type == "analysis":
                prompt += "복잡한 제주어 표현의 특성을 심층적으로 분석하겠습니다.\n"
                prompt += "1. 제주어 표현의 언어학적 특성 분석하기\n"
                prompt += "2. 표준어와의 음운적/문법적 차이점 식별하기\n"
                prompt += "3. 제주어 표현의 역사적/문화적 맥락 고려하기\n"
                prompt += "4. 제주어 패턴과 규칙성 도출하기\n"
                prompt += "5. 분석 결과 종합 및 검증하기\n\n"
            else:
                prompt += "복잡한 제주어 관련 질문에 심층적으로 답변하겠습니다.\n"
                prompt += "1. 질문의 핵심 의도 분석하기\n"
                prompt += "2. 제주어 관련 정보 종합하기\n"
                prompt += "3. 문화적/역사적 맥락 고려하기\n"
                prompt += "4. 관련된 제주 지역 정보 파악하기\n"
                prompt += "5. 종합적인 답변 구성 및 검증하기\n\n"
        
        # Contextual Retrieval 결과 추가
        if retrieval_response:
            # 검색 결과 추가
            prompt += "참고할 수 있는 관련 정보는 다음과 같습니다:\n"
            
            # 상위 검색 결과 추가
            for i, result in enumerate(retrieval_response['results'][:3], 1):
                prompt += f"참고 정보 {i}:\n"
                prompt += f"- 제주어: {result['form']}\n"
                if 'standard_form' in result and result['standard_form'] != result['form']:
                    prompt += f"- 표준어: {result['standard_form']}\n"
                if 'dialect_ratio' in result:
                    prompt += f"- 방언 비율: {result['dialect_ratio']:.2%}\n"
                prompt += "\n"
            
            # 인사이트 추가
            if 'insights' in retrieval_response and retrieval_response['insights']:
                insights = retrieval_response['insights']
                
                # 제주어 단어 번역 정보 추가
                if 'dialect_words' in insights and insights['dialect_words']:
                    prompt += "제주어-표준어 매핑:\n"
                    for dialect, standard in insights['dialect_words'][:5]:
                        prompt += f"- '{dialect}' → '{standard}'\n"
                    prompt += "\n"
                
                # 패턴 정보 추가
                if 'dialect_patterns' in insights and query_type in ['analysis', 'question']:
                    patterns = insights['dialect_patterns']
                    if 'most_common_suffixes' in patterns and patterns['most_common_suffixes']:
                        prompt += "자주 사용되는 제주어 접미사:\n"
                        for suffix, count in patterns['most_common_suffixes'][:3]:
                            prompt += f"- '{suffix}' (빈도: {count})\n"
                        prompt += "\n"
                    
                    if 'most_common_prefixes' in patterns and patterns['most_common_prefixes']:
                        prompt += "자주 사용되는 제주어 접두사:\n"
                        for prefix, count in patterns['most_common_prefixes'][:3]:
                            prompt += f"- '{prefix}' (빈도: {count})\n"
                        prompt += "\n"
        
        # 사고 과정 요청
        prompt += "이제 단계적으로 생각하며 답변을 작성하겠습니다:\n"
        
        return prompt
    
    def generate_verification_prompt(self, query, answer):
        """
        자체 검증을 위한 프롬프트를 생성합니다.
        
        Args:
            query (str): 사용자 쿼리
            answer (str): 생성된 답변
        
        Returns:
            str: 검증 프롬프트
        """
        prompt = f"""제주어 전문가로서 다음 질문과 답변의 정확성을 검증하겠습니다.

질문: {query}

답변: {answer}

단계적으로 검증을 수행하겠습니다:
1. 답변에 포함된 제주어 표현 정확성 검증
2. 제주어-표준어 번역의 정확성 검증
3. 문맥적 의미 전달의 정확성 검증
4. 문화적/지역적 정보의 정확성 검증
5. 개선이 필요한 부분 식별

검증 결과:
"""
        return prompt
    
    def cot_reasoning(self, query, retrieval_response=None):
        """
        Chain-of-Thought 추론을 수행합니다.
        """
        if not self.anthropic_client:
            logger.warning("Voyage 모델이 준비되지 않았습니다.")
            return "모델이 준비되지 않아 추론을 수행할 수 없습니다."
        
        # 프롬프트 생성
        prompt = self.generate_cot_prompt(query, retrieval_response)
        
        # Anthropic API 호출
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=4096,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            answer = response.content[0].text
            
            logger.info(f"Voyage CoT 추론 수행 완료: {len(answer.split())} 토큰 생성")
            return answer
        except Exception as e:
            logger.error(f"Voyage CoT 추론 실패: {e}")
            return f"추론 과정에서 오류가 발생했습니다: {str(e)}"
    
    def self_verification(self, query, answer):
        """
        자체 검증을 수행합니다.
        
        Args:
            query (str): 사용자 쿼리
            answer (str): 생성된 답변
        
        Returns:
            str: 검증 및 개선된 답변
        """
        if not self.anthropic_client:
            logger.warning("Voyage 모델이 준비되지 않아 자체 검증을 수행할 수 없습니다.")
            return answer
        
        # 검증 프롬프트 생성
        verification_prompt = self.generate_verification_prompt(query, answer)
        
        # 검증 수행
        try:
            verification_response = self.anthropic_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=4096,
                messages=[
                    {"role": "user", "content": verification_prompt}
                ]
            )
            
            verification_result = verification_response.content[0].text
            
            logger.info("자체 검증 수행 완료")
            
            # 검증 결과 분석
            if "문제 없음" in verification_result or "정확함" in verification_result:
                return answer
            else:
                # 오류 수정 프롬프트 생성
                correction_prompt = f"""제주어 전문가로서 다음 질문에 대한 답변을 개선하겠습니다.

질문: {query}

원래 답변: {answer}

검증 결과: {verification_result}

개선된 답변:
"""
                
                # 개선된 답변 생성
                correction_response = self.anthropic_client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=4096,
                    messages=[
                        {"role": "user", "content": correction_prompt}
                    ]
                )
                
                improved_answer = correction_response.content[0].text
                
                logger.info("답변 개선 완료")
                
                return improved_answer
        except Exception as e:
            logger.error(f"자체 검증 실패: {e}")
            return answer

class JejuDialectHybridSystem:
    """
    제주어 방언 하이브리드 시스템
    HDLCoRe 논문의 접근법을 제주어 도메인에 적용한 통합 시스템
    """
    def __init__(self, data_path=None):
        self.processor = JejuDialectProcessor(data_path)
        self.rag_system = None
        self.contextual_rag_system = None
        self.cot_system = None
    
    def initialize_systems(self, embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 
                          llm_model_name="anthropic/voyage",
                          use_contextual_rag=True):
        """
        RAG 및 CoT 시스템을 초기화합니다.
        
        Args:
            embedding_model_name (str): 임베딩 모델 이름
            llm_model_name (str): LLM 모델 이름
            use_contextual_rag (bool): Contextual RAG 사용 여부
        """
        # RAG 시스템 초기화
        if use_contextual_rag:
            self.contextual_rag_system = JejuDialectContextualRAG(self.processor, embedding_model_name)
            
            # 이미 구축된 데이터베이스가 있는지 확인
            rag_path = "jeju_dialect_contextual_rag.pkl"
            if os.path.exists(rag_path):
                logger.info(f"기존 Contextual RAG 시스템 로드 시도: {rag_path}")
                if not self.contextual_rag_system.load(rag_path):
                    logger.info("기존 Contextual RAG 시스템 로드 실패, 새로 구축합니다.")
                    self.contextual_rag_system.build_database()
                    self.contextual_rag_system.save(rag_path)
            else:
                logger.info("Contextual RAG 시스템 새로 구축합니다.")
                self.contextual_rag_system.build_database()
                self.contextual_rag_system.save(rag_path)
        else:
            self.rag_system = JejuDialectRAG(self.processor, embedding_model_name)
            
            # 이미 구축된 데이터베이스가 있는지 확인
            rag_path = "jeju_dialect_rag.pkl"
            if os.path.exists(rag_path):
                logger.info(f"기존 RAG 시스템 로드 시도: {rag_path}")
                if not self.rag_system.load(rag_path):
                    logger.info("기존 RAG 시스템 로드 실패, 새로 구축합니다.")
                    self.rag_system.build_database()
                    self.rag_system.save(rag_path)
            else:
                logger.info("RAG 시스템 새로 구축합니다.")
                self.rag_system.build_database()
                self.rag_system.save(rag_path)
        
        # CoT 시스템 초기화
        self.cot_system = JejuDialectCoT(self.processor, llm_model_name)
    
    def process_query(self, query, use_rag=True, use_cot=True, conversation_history="", use_contextual_rag=True):
        """
        사용자 쿼리를 처리합니다.
        
        Args:
            query (str): 사용자 쿼리
            use_rag (bool): RAG 시스템 사용 여부
            use_cot (bool): CoT 시스템 사용 여부
            conversation_history (str): 대화 기록
            use_contextual_rag (bool): Contextual RAG 사용 여부
        
        Returns:
            dict: 처리 결과
        """
        result = {
            "query": query,
            "answer": "",
            "rag_results": None,
            "cot_used": use_cot
        }
        
        # 1. RAG 검색 수행 (사용 설정된 경우)
        if use_rag:
            if use_contextual_rag and self.contextual_rag_system:
                # Contextual RAG 사용
                retrieval_response = self.contextual_rag_system.retrieve(query, conversation_history=conversation_history)
                result["rag_results"] = retrieval_response
                logger.info(f"Contextual RAG 검색 완료: {len(retrieval_response['results'])}개 결과")
            elif self.rag_system:
                # 기본 RAG 사용
                rag_results = self.rag_system.retrieve(query)
                result["rag_results"] = rag_results
                logger.info(f"RAG 검색 완료: {len(rag_results)}개 결과")
            else:
                retrieval_response = None
        else:
            retrieval_response = None
        
        # 2. CoT 추론 수행 (사용 설정된 경우)
        if use_cot and self.cot_system:
            answer = self.cot_system.cot_reasoning(query, result["rag_results"])
            result["answer"] = answer
            logger.info("CoT 추론 완료")
        else:
            # 간단한 규칙 기반 응답 (CoT 사용하지 않는 경우)
            answer = self._rule_based_response(query, result["rag_results"])
            result["answer"] = answer
            logger.info("규칙 기반 응답 생성 완료")
        
        return result
    
    def _rule_based_response(self, query, rag_results):
        """
        규칙 기반 응답을 생성합니다. (CoT를 사용하지 않는 경우)
        
        Args:
            query (str): 사용자 쿼리
            rag_results (dict/list): RAG 검색 결과
        
        Returns:
            str: 생성된 응답
        """
        response = f"질문: {query}\n\n"
        
        # Contextual RAG 결과인 경우
        if isinstance(rag_results, dict) and 'results' in rag_results and 'insights' in rag_results:
            rag_results_list = rag_results['results']
            insights = rag_results['insights']
            
            # 번역 요청 감지
            if "번역" in query or "뜻" in query:
                response += "제주어 번역 결과:\n\n"
                
                if 'dialect_words' in insights and insights['dialect_words']:
                    for word, std in insights['dialect_words']:
                        response += f"{word} → {std}\n"
                else:
                    # 쿼리 내 제주어 단어 찾기
                    query_words = query.split()
                    translated_words = []
                    
                    for word in query_words:
                        if word in self.processor.dialect_dict:
                            translated_words.append(f"{word} → {self.processor.dialect_dict[word]}")
                    
                    if translated_words:
                        response += "\n".join(translated_words)
                    else:
                        response += "번역할 제주어 단어를 찾지 못했습니다."
            
            # 분석 요청 감지
            elif "분석" in query or "패턴" in query or "특징" in query:
                response += "제주어 패턴 분석 결과:\n\n"
                
                if 'dialect_patterns' in insights and insights['dialect_patterns']:
                    patterns = insights['dialect_patterns']
                    
                    if 'most_common_suffixes' in patterns and patterns['most_common_suffixes']:
                        response += "자주 사용되는 제주어 접미사:\n"
                        for suffix, count in patterns['most_common_suffixes'][:5]:
                            response += f"- '{suffix}' (빈도: {count})\n"
                        response += "\n"
                    
                    if 'most_common_prefixes' in patterns and patterns['most_common_prefixes']:
                        response += "자주 사용되는 제주어 접두사:\n"
                        for prefix, count in patterns['most_common_prefixes'][:5]:
                            response += f"- '{prefix}' (빈도: {count})\n"
                        response += "\n"
                    
                    if 'most_common_replacements' in patterns and patterns['most_common_replacements']:
                        response += "자주 사용되는 제주어 문자 변형:\n"
                        for (std, dialect), count in patterns['most_common_replacements'][:5]:
                            response += f"- '{std}' → '{dialect}' (빈도: {count})\n"
                else:
                    response += "제주어 패턴을 분석하는 중 오류가 발생했습니다."
            
            # 일반 응답
            else:
                response += "관련된 제주어 정보:\n\n"
                
                for i, result in enumerate(rag_results_list[:5], 1):
                    response += f"{i}. 제주어: {result['form']}\n"
                    if 'standard_form' in result and result['standard_form'] != result['form']:
                        response += f"   표준어: {result['standard_form']}\n"
                    if 'dialect_ratio' in result:
                        response += f"   방언 비율: {result['dialect_ratio']:.2%}\n"
                    response += "\n"
                
                # 추가 인사이트 포함
                if 'summary' in insights and insights['summary']:
                    response += f"요약: {insights['summary']}\n"
        
        # 기본 RAG 결과인 경우
        elif isinstance(rag_results, list):
            # 번역 요청 감지
            if "번역" in query or "뜻" in query:
                response += "제주어 번역 결과:\n\n"
                
                # 쿼리 내 제주어 단어 찾기
                query_words = query.split()
                translated_words = []
                
                for word in query_words:
                    if word in self.processor.dialect_dict:
                        translated_words.append(f"{word} → {self.processor.dialect_dict[word]}")
                
                if translated_words:
                    response += "\n".join(translated_words)
                else:
                    response += "번역할 제주어 단어를 찾지 못했습니다."
            
            # RAG 결과 추가
            elif rag_results:
                response += "관련된 제주어 정보:\n\n"
                
                for i, result in enumerate(rag_results, 1):
                    response += f"{i}. 제주어: {result['form']}\n"
                    response += f"   표준어: {result['standard_form']}\n\n"
            
            # 일반 정보 제공
            else:
                response += "제주어에 관한 정보:\n\n"
                
                # 제주어 패턴 정보
                patterns = self.processor.get_pattern_analysis()
                
                if patterns:
                    response += "자주 사용되는 제주어 접미사:\n"
                    for suffix, count in patterns['most_common_suffixes'][:3]:
                        response += f"- '{suffix}' (빈도: {count})\n"
                    
                    response += "\n자주 사용되는 제주어 접두사:\n"
                    for prefix, count in patterns['most_common_prefixes'][:3]:
                        response += f"- '{prefix}' (빈도: {count})\n"
        
        # RAG 결과가 없는 경우
        else:
            response += "제주어에 관한 기본 정보:\n\n"
            response += "제주어 관련 검색을 수행할 수 없습니다. 더 구체적인 질문을 해주세요."
        
        return response

class JejuDialectContextualRAG:
    """
    제주어 방언 Contextual Retrieval 시스템
    논문에 기반한 다중 검색 전략 및 하이브리드 검색 구현
    """
    def __init__(self, processor, embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.processor = processor
        self.database = []
        self.embedding_model = None
        self.cross_encoder = None
        self.embedding_model_name = embedding_model_name
        
        # BM25 검색을 위한 변수
        self.bm25_tokenized_corpus = []
        self.bm25_model = None
        
        # TF-IDF 검색을 위한 변수
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        
        # 임베딩 모델 초기화
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
            logger.info(f"임베딩 모델 로드 성공: {embedding_model_name}")
        except Exception as e:
            logger.error(f"임베딩 모델 로드 실패: {e}")
        
        # 크로스 인코더 초기화 (재순위 지정용)
        try:
            self.cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
            logger.info("크로스 인코더 로드 성공")
        except Exception as e:
            logger.error(f"크로스 인코더 로드 실패: {e}")
    
    def build_database(self):
        """
        다중 검색을 위한 데이터베이스를 구축합니다.
        """
        if not self.processor.utterances:
            logger.warning("데이터베이스 구축을 위한 발화 데이터가 없습니다.")
            return
        
        # 발화를 데이터베이스 항목으로 변환
        database_items = []
        tokenized_corpus = []
        
        for utterance in self.processor.utterances:
            # 발화 기본 정보
            form = utterance.get('form', '')
            item = {
                "id": utterance.get('id', ''),
                "speaker_id": utterance.get('speaker_id', ''),
                "form": form,
                "standard_form": utterance.get('standard_form', form),
                "dialect_form": utterance.get('dialect_form', form)
            }
            
            # 키 구성요소 추출
            components = self.processor.extract_key_components(item["form"])
            item.update(components)
            
            # 방언 비율 계산
            if 'eojeolList' in utterance:
                eojeols = utterance['eojeolList']
                total_words = len(eojeols)
                dialect_words = sum(1 for eojeol in eojeols if eojeol.get('isDialect', False))
                item["dialect_ratio"] = dialect_words / total_words if total_words > 0 else 0
            
            database_items.append(item)
            
            # BM25를 위한 토큰화
            tokenized_doc = form.split()
            tokenized_corpus.append(tokenized_doc)
        
        self.database = database_items
        self.bm25_tokenized_corpus = tokenized_corpus
        
        # BM25 모델 초기화
        self.bm25_model = BM25Okapi(self.bm25_tokenized_corpus)
        
        # TF-IDF 행렬 계산
        documents = [item["form"] for item in self.database]
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        
        logger.info(f"다중 검색 데이터베이스 구축 완료: {len(self.database)}개 항목")
        
        # 임베딩 생성
        self._create_embeddings()
    
    def _create_embeddings(self):
        """
        데이터베이스 항목에 대한 임베딩을 생성합니다.
        """
        if not self.embedding_model:
            logger.warning("임베딩 모델이 초기화되지 않았습니다.")
            return
        
        # 고수준, 저수준, 구조적 구성요소에 대한 임베딩 생성
        self.database_embeddings = {
            "high_level": [],
            "low_level": [],
            "structural": [],
            "form": []  # 전체 발화에 대한 임베딩 추가
        }
        
        high_level_texts = [item["high_level"] for item in self.database]
        low_level_texts = [item["low_level"] for item in self.database]
        structural_texts = [item["structural"] for item in self.database]
        form_texts = [item["form"] for item in self.database]
        
        # 임베딩 계산
        self.database_embeddings["high_level"] = self.embedding_model.encode(high_level_texts)
        self.database_embeddings["low_level"] = self.embedding_model.encode(low_level_texts)
        self.database_embeddings["structural"] = self.embedding_model.encode(structural_texts)
        self.database_embeddings["form"] = self.embedding_model.encode(form_texts)
        
        logger.info("데이터베이스 임베딩 생성 완료")
    
    def _analyze_query_intent(self, query, conversation_history=""):
        """
        쿼리의 의도를 분석합니다.
        논문에서 언급된 질의 의도 분석 개념 적용
        
        Args:
            query (str): 사용자 쿼리
            conversation_history (str): 이전 대화 기록
        
        Returns:
            dict: 분석된 의도 정보
        """
        # 제주어 방언 관련 키워드 패턴
        translation_pattern = re.compile(r'(번역|뜻|의미|표준어로|제주어로)')
        analysis_pattern = re.compile(r'(분석|특징|패턴|차이)')
        question_pattern = re.compile(r'(무엇|언제|어디|누구|어떻게|왜|어떤)')
        
        # 제주어 단어 비율 계산
        dialect_words = 0
        total_words = len(query.split())
        for word in query.split():
            if word in self.processor.dialect_dict:
                dialect_words += 1
        dialect_ratio = dialect_words / total_words if total_words > 0 else 0
        
        # 패턴 매칭
        is_translation = bool(translation_pattern.search(query))
        is_analysis = bool(analysis_pattern.search(query))
        is_question = bool(question_pattern.search(query))
        
        # 의도 분석
        intent = {
            "dialect_ratio": dialect_ratio,
            "is_translation": is_translation,
            "is_analysis": is_analysis,
            "is_question": is_question
        }
        
        # 주요 의도 결정
        if dialect_ratio > 0.3:
            intent["primary"] = "translation"
        elif is_translation:
            intent["primary"] = "translation"
        elif is_analysis:
            intent["primary"] = "analysis"
        elif is_question:
            intent["primary"] = "question"
        else:
            intent["primary"] = "general"
        
        return intent
    
    def _decompose_query(self, query, intent):
        """
        쿼리를 분해하여 하위 쿼리를 생성합니다.
        논문에서 언급된 질의 분해 개념 적용
        
        Args:
            query (str): 원본 쿼리
            intent (dict): 분석된 의도
        
        Returns:
            list: 하위 쿼리 목록
        """
        sub_queries = []
        words = query.split()
        
        # 1. 원본 쿼리 추가
        sub_queries.append(query)
        
        # 2. 제주어 단어만 추출
        dialect_words = [word for word in words if word in self.processor.dialect_dict]
        if dialect_words:
            dialect_query = " ".join(dialect_words)
            sub_queries.append(dialect_query)
        
        # 3. 의도 기반 서브쿼리 생성
        if intent["primary"] == "translation" and dialect_words:
            for word in dialect_words:
                sub_queries.append(f"{word} 뜻")
        
        elif intent["primary"] == "analysis":
            sub_queries.append("제주어 특징")
            sub_queries.append("제주어 패턴")
        
        # 4. 대표적인 n-gram 추출 (2~3 단어 조합)
        if len(words) >= 2:
            for i in range(len(words) - 1):
                bigram = words[i] + " " + words[i+1]
                sub_queries.append(bigram)
        
        if len(words) >= 3:
            for i in range(len(words) - 2):
                trigram = words[i] + " " + words[i+1] + " " + words[i+2]
                sub_queries.append(trigram)
        
        # 중복 제거
        sub_queries = list(set(sub_queries))
        
        return sub_queries
    
    def _hybrid_retrieval(self, query, top_k=10):
        """
        하이브리드 검색 수행 (임베딩 + BM25 + TF-IDF)
        논문에서 언급된 다중 검색 전략 적용
        
        Args:
            query (str): 검색 쿼리
            top_k (int): 반환할 결과 수
        
        Returns:
            list: 검색 결과 목록
        """
        # 1. 임베딩 기반 검색 (고수준 구성요소)
        query_embedding = self.embedding_model.encode(query)
        high_level_similarities = cosine_similarity(
            [query_embedding],
            self.database_embeddings["form"]
        )[0]
        
        # 상위 결과 인덱스 추출
        embedding_top_indices = np.argsort(high_level_similarities)[-top_k:][::-1]
        embedding_scores = {idx: high_level_similarities[idx] for idx in embedding_top_indices}
        
        # 2. BM25 검색
        tokenized_query = query.split()
        bm25_scores = self.bm25_model.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[-top_k:][::-1]
        bm25_top_scores = {idx: bm25_scores[idx] for idx in bm25_top_indices}
        
        # 3. TF-IDF 검색
        tfidf_query = self.tfidf_vectorizer.transform([query])
        tfidf_similarities = cosine_similarity(tfidf_query, self.tfidf_matrix)[0]
        tfidf_top_indices = np.argsort(tfidf_similarities)[-top_k:][::-1]
        tfidf_top_scores = {idx: tfidf_similarities[idx] for idx in tfidf_top_indices}
        
        # 결과 통합 및 스코어 정규화
        all_indices = set(list(embedding_top_indices) + list(bm25_top_indices) + list(tfidf_top_indices))
        combined_scores = {}
        
        # 정규화 함수
        def normalize(scores):
            if not scores:
                return {}
            max_score = max(scores.values())
            min_score = min(scores.values())
            if max_score == min_score:
                return {idx: 1.0 for idx in scores}
            return {idx: (score - min_score) / (max_score - min_score) for idx, score in scores.items()}
        
        # 점수 정규화
        embedding_scores_norm = normalize(embedding_scores)
        bm25_scores_norm = normalize(bm25_top_scores)
        tfidf_scores_norm = normalize(tfidf_top_scores)
        
        # 가중치 합산 (여기서는 임베딩에 더 높은 가중치)
        for idx in all_indices:
            combined_scores[idx] = (
                embedding_scores_norm.get(idx, 0) * 0.5 +
                bm25_scores_norm.get(idx, 0) * 0.3 +
                tfidf_scores_norm.get(idx, 0) * 0.2
            )
        
        # 상위 결과 선택
        top_indices = sorted(combined_scores.keys(), key=lambda idx: combined_scores[idx], reverse=True)[:top_k]
        
        # 결과 구성
        results = []
        for idx in top_indices:
            item = self.database[idx].copy()
            item['score'] = combined_scores[idx]
            results.append(item)
        
        return results
    
    def _assess_relevance(self, query, intent, results):
        """
        검색 결과의 관련성을 평가하고 재순위화합니다.
        논문에서 언급된 문서 관련성 평가 개념 적용
        
        Args:
            query (str): 원본 쿼리
            intent (dict): 분석된 의도
            results (list): 검색 결과 목록
        
        Returns:
            list: 관련성 점수가 재계산된 결과 목록
        """
        if not self.cross_encoder or not results:
            return results
        
        # 쿼리-문서 쌍 생성
        pairs = [(query, item["form"]) for item in results]
        
        # 크로스 인코더로 관련성 점수 계산
        cross_scores = self.cross_encoder.predict(pairs)
        
        # 어떤 의도인지에 따라 점수 조정
        if intent["primary"] == "translation":
            # 제주어 단어가 포함된 결과 점수 가중치
            dialect_words = [word for word in query.split() if word in self.processor.dialect_dict]
            if dialect_words:
                for i, item in enumerate(results):
                    for word in dialect_words:
                        if word in item["form"]:
                            cross_scores[i] *= 1.2  # 가중치 부여
        
        elif intent["primary"] == "analysis":
            # 방언 비율이 높은 결과 점수 가중치
            for i, item in enumerate(results):
                if item.get("dialect_ratio", 0) > 0.5:
                    cross_scores[i] *= 1.1  # 가중치 부여
        
        # 점수를 결과에 추가하고 정렬
        for i, item in enumerate(results):
            item["relevance_score"] = float(cross_scores[i])
        
        # 관련성 점수로 정렬
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return results
    
    def _extract_insights(self, query, intent, results, top_n=5):
        """
        검색 결과에서 인사이트를 추출합니다.
        논문에서 언급된 인사이트 추출 개념 적용
        
        Args:
            query (str): 원본 쿼리
            intent (dict): 분석된 의도
            results (list): 검색 결과 목록
            top_n (int): 사용할 상위 결과 수
        
        Returns:
            dict: 추출된 인사이트
        """
        insights = {
            "dialect_words": [],
            "dialect_patterns": [],
            "summary": ""
        }
        
        # 상위 결과만 사용
        top_results = results[:top_n]
        
        # 1. 제주어 단어 추출
        for item in top_results:
            form = item["form"]
            words = form.split()
            for word in words:
                if word in self.processor.dialect_dict:
                    standard = self.processor.dialect_dict[word]
                    insights["dialect_words"].append((word, standard))
        
        # 중복 제거
        insights["dialect_words"] = list(set(insights["dialect_words"]))
        
        # 2. 패턴 분석 추가
        if intent["primary"] == "analysis" or "패턴" in query or "특징" in query:
            patterns = self.processor.get_pattern_analysis()
            insights["dialect_patterns"] = patterns
        
        # 3. 요약 생성
        if intent["primary"] == "translation":
            dialect_words_str = ", ".join([f"{word}→{std}" for word, std in insights["dialect_words"][:5]])
            insights["summary"] = f"제주어 번역: {dialect_words_str}"
        elif intent["primary"] == "analysis":
            insights["summary"] = "제주어 패턴 분석 결과입니다."
        else:
            insights["summary"] = "제주어 관련 검색 결과입니다."
        
        return insights
    
    def retrieve(self, query, top_k=10, conversation_history=""):
        """
        Contextual Retrieval 수행
        논문에서 제시한 단계별 검색 프로세스 구현
        
        Args:
            query (str): 사용자 쿼리
            top_k (int): 반환할 결과 수
            conversation_history (str): 이전 대화 기록
        
        Returns:
            dict: 검색 결과 및 생성된 응답
        """
        # 1. 쿼리 의도 분석
        intent = self._analyze_query_intent(query, conversation_history)
        logger.info(f"쿼리 의도 분석 완료: {intent['primary']}")
        
        # 2. 쿼리 분해
        sub_queries = self._decompose_query(query, intent)
        logger.info(f"쿼리 분해 완료: {len(sub_queries)}개 하위 쿼리 생성")
        
        # 3. 하이브리드 검색 수행
        all_results = []
        for sub_query in sub_queries:
            results = self._hybrid_retrieval(sub_query, top_k=5)
            all_results.extend(results)
        
        # 중복 제거
        unique_results = []
        seen_ids = set()
        for result in all_results:
            if result["id"] not in seen_ids:
                unique_results.append(result)
                seen_ids.add(result["id"])
        
        # 4. 관련성 평가 및 재순위화
        ranked_results = self._assess_relevance(query, intent, unique_results)
        top_results = ranked_results[:top_k]
        logger.info(f"관련성 평가 및 재순위화 완료: {len(top_results)}개 결과")
        
        # 5. 인사이트 추출
        insights = self._extract_insights(query, intent, top_results)
        logger.info("인사이트 추출 완료")
        
        # 최종 응답 구성
        response = {
            "query": query,
            "intent": intent,
            "results": top_results,
            "insights": insights
        }
        
        return response
    
    def save(self, path="jeju_dialect_rag.pkl"):
        """
        RAG 시스템을 파일로 저장합니다.
        
        Args:
            path (str): 저장 경로
        """
        save_data = {
            "database": self.database,
            "database_embeddings": self.database_embeddings,
            "embedding_model_name": self.embedding_model_name,
            # BM25와 TF-IDF 모델은 직렬화가 어려울 수 있으므로 제외
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Contextual RAG 시스템 저장 완료: {path}")
    
    def load(self, path="jeju_dialect_rag.pkl"):
        """
        RAG 시스템을 파일에서 로드합니다.
        
        Args:
            path (str): 로드 경로
        
        Returns:
            bool: 로드 성공 여부
        """
        try:
            with open(path, 'rb') as f:
                save_data = pickle.load(f)
            
            self.database = save_data["database"]
            self.database_embeddings = save_data["database_embeddings"]
            
            # 모델 이름이 다른 경우 재로드
            if save_data["embedding_model_name"] != self.embedding_model_name:
                self.embedding_model_name = save_data["embedding_model_name"]
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            # BM25와 TF-IDF 모델 재구축
            documents = [item["form"] for item in self.database]
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
            
            self.bm25_tokenized_corpus = [doc.split() for doc in documents]
            self.bm25_model = BM25Okapi(self.bm25_tokenized_corpus)
            
            logger.info(f"Contextual RAG 시스템 로드 성공: {path}")
            return True
        except Exception as e:
            logger.error(f"Contextual RAG 시스템 로드 실패: {e}")
            return False

def main():
    """
    메인 함수
    """
    # 데이터 경로 설정
    data_path = "DZHF20002075.json"
    
    # 하이브리드 시스템 초기화
    system = JejuDialectHybridSystem(data_path)
    
    # RAG 및 CoT 시스템 초기화
    system.initialize_systems(
        embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        llm_model_name="anthropic/voyage",
        use_contextual_rag=True
    )
    
    # 테스트 쿼리
    test_queries = [
        "게난은 표준어로 무슨 뜻인가요?",
        "제주어에서 자주 사용되는 접미사는 무엇인가요?",
        "게난 일 학년하고 얘네가 지금 복학했으니까 당연히 라는 문장을 번역해주세요",
        "제주어와 표준어의 가장 큰 차이점은 무엇인가요?"
    ]
    
    # 대화 기록 초기화
    conversation_history = ""
    
    # 테스트 쿼리 처리
    for i, query in enumerate(test_queries, 1):
        print(f"\n===== 테스트 쿼리 {i} =====")
        print(f"쿼리: {query}")
        
        result = system.process_query(
            query, 
            use_rag=True, 
            use_cot=True,
            conversation_history=conversation_history,
            use_contextual_rag=True
        )
        
        # 대화 기록 업데이트
        conversation_history += f"사용자: {query}\n시스템: {result['answer']}\n\n"
        
        print("\n[답변]")
        print(result["answer"])
        
        if result["rag_results"] and isinstance(result["rag_results"], dict) and 'results' in result["rag_results"]:
            print("\n[관련 Contextual RAG 검색 결과]")
            for j, rag_result in enumerate(result["rag_results"]["results"][:2], 1):
                print(f"{j}. 제주어: {rag_result['form']}")
                if 'standard_form' in rag_result and rag_result['standard_form'] != rag_result['form']:
                    print(f"   표준어: {rag_result['standard_form']}")
        elif result["rag_results"] and isinstance(result["rag_results"], list):
            print("\n[관련 RAG 검색 결과]")
            for j, rag_result in enumerate(result["rag_results"][:2], 1):
                print(f"{j}. 제주어: {rag_result['form']}")
                print(f"   표준어: {rag_result['standard_form']}")
        
        print("=" * 50)

if __name__ == "__main__":
    main()