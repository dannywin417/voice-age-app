# api/analyze.py

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import io
from pydub import AudioSegment
import random
import traceback
import hashlib

API_VERSION = "1.3.0"

app = FastAPI()

# CORS (필요 시 도메인 추가)
# origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
origins = ["*"] # 🚨 이렇게 모든 주소를 허용하도록 변경합니다. (테스트용)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 제한/가드 상수 ---
MIN_SEC = 0.5
MAX_SEC = 12
MAX_BYTES = 10 * 1024 * 1024  # 10MB

# --- 데이터 풀 ---
voice_analysis_data = {
    "age_groups": {
        "10s": {"range": "10대", "humor": ["목소리에서 '엄마 용돈 올려달라'는 간절함이 느껴져요! 🌸", "라면 끓이는 소리만 들어도 달려올 것 같은 목소리! ⚡", "밤 12시에 '숙제 언제 하지?' 하는 목소리네요! 📱", "청춘 드라마에서 '야, 너 좋아해' 고백할 목소리! 🎭", "새로 나온 줄임말을 일주일 만에 마스터할 것 같아요!", "에너지 드링크가 목소리로 변한 느낌!"]},
        "20s_early": {"range": "20대 초반", "humor": ["대학 과제 마감 2시간 전의 절망과 희망이 공존하는 목소리! 🎓", "밤새 팀플하고 '이번엔 진짜 A+ 받을 거야' 하는 목소리!", "MT에서 '우리 과 최고!' 외칠 목소리에요!", "'오늘 뭐 먹지?'가 인생 최대 고민인 목소리!", "개강파티 주최자 목소리네요!", "친구 번호 물어봐달라고 부탁받을 목소리!"]},
        "20s_late": {"range": "20대 후반", "humor": ["이제 막 '어른'이라는 가면을 쓰기 시작한 목소리! 🍺", "퇴근길에 '오늘도 고생했다' 혼잣말할 목소리네요!", "첫 월급으로 '이제야 사람 됐다' 느끼는 목소리!", "주말 약속 없으면 인싸 자격 박탈당할 것 같은 목소리!", "연애할 때 가장 설레지만 현실은 솔로인 목소리 💕", "독립 후 '집에서 속옷만 입고 다니는 자유'를 만끽하는 목소리!"]},
        "30s_early": {"range": "30대 초반", "humor": ["안정감은 있는데 여전히 게임 밤새는 목소리예요.", "넷플릭스 정주행이 최고의 힐링이라고 확신하는 목소리!", "'아, 허리야...' 첫 신음소리를 낸 목소리!", "회사에서 '믿고 맡길 수 있는' 목소리 💼", "결혼식 축사에서 웃음과 감동을 동시에 줄 목소리!", "커피 없으면 좀비가 되는 목소리 ☕"]},
        "30s_late": {"range": "30대 후반", "humor": ["깊이는 있는데 아직 유튜브 알고리즘에 당하는 목소리! 💰", "재테크 유튜브 보면서 '나도 부자 될 수 있어' 하는 목소리!", "육아 현실에 치여도 아이 앞에선 천사가 되는 목소리 👶", "캠핑 가서 '자연이 최고야' 하지만 와이파이 찾는 목소리!", "인생 황금기라지만 체력은 이미 하향곡선인 목소리! ✨", "'집이 천국'이라는 진리를 깨달은 목소리!"]},
        "40s_early": {"range": "40대 초반", "humor": ["편안하지만 갑자기 '요즘 애들은...' 하고 싶어지는 목소리!", "와인 마시면서 '인생을 논하고' 싶어지는 목소리 🍷", "후배들에게 밥 사주면서 '내 젊었을 때는' 시전할 목소리!", "아이 숙제 도와주다가 '이게 뭐야?' 할 목소리 📝", "경험담이 레전드가 된 목소리 📚", "골프 치면서 '스트레스 푸는' 목소리!"]},
        "40s_late": {"range": "40대 후반", "humor": ["지혜롭지만 아직 스마트폰 기능을 다 모르는 목소리!", "'내가 너 때는...' 전설의 시작을 알리는 목소리!", "인생의 단맛을 아는 동시에 쓴맛도 아는 목소리!", "다큐 보면서 '역시 옛날이 좋았어' 할 목소리!", "가족여행 계획 세우는 게 취미가 된 목소리!", "친구 모임에서 '건강이 최고야' 외치는 목소리!"]},
        "50s_plus": {"range": "50대 이상", "humor": ["모든 것을 다 겪어본 '인생 고수'의 여유로운 목소리!", "차 한 잔에 '인생 철학'을 담아낼 수 있는 목소리 🍵", "'이제야 진짜 내 인생이 시작이야' 하는 목소리 🎭", "손자 손녀에게 '옛날에 할아버지는...' 시전하는 목소리!", "등산복이 일상복이 된 목소리 🏔️", "텃밭에서 '내가 기른 배추가 최고야' 하는 목소리!"]},
    },
    "animalTypes": [
        {"id": "cat", "type": "고양이상", "emoji": "🐱", "desc": "츤데레의 완성체, 관심없는 척 하지만 사실 관종"},
        {"id": "dog", "type": "강아지상", "emoji": "🐶", "desc": "세상 모든 사람이 좋은 사람일 거라고 믿는 순수함"},
        {"id": "bear", "type": "곰상", "emoji": "🐻", "desc": "포근한 인간 담요, 안기고 싶게 만드는 마성의 체질"},
        {"id": "fox", "type": "여우상", "emoji": "🦊", "desc": "계산기보다 빠른 두뇌, 눈빛만으로 사람 홀리는 마법사"},
        {"id": "hamster", "type": "햄스터상", "emoji": "🐹", "desc": "입에 음식 가득 넣고도 귀여운 반칙급 외모"},
        {"id": "lion", "type": "사자상", "emoji": "🦁", "desc": "가만히 있어도 포스 폭발, 천상천하 유아독존"},
        {"id": "rabbit", "type": "토끼상", "emoji": "🐰", "desc": "깜찍함으로 세상을 정복하는 중, 보호본능 자극 전문가"},
        {"id": "wolf", "type": "늑대상", "emoji": "🐺", "desc": "야성미 철철 흘러넘치는 매력, 길들여지지 않는 자유로운 영혼"}
    ],
    "personalityTypes": [
        {"id": "leader", "type": "타고난 리더형", "emoji": "👑", "color": "#f59e0b"},
        {"id": "emotional", "type": "따뜻한 감성형", "emoji": "💝", "color": "#ec4899"},
        {"id": "maker", "type": "분위기 메이커형", "emoji": "🎉", "color": "#8b5cf6"},
        {"id": "stable", "type": "깊고 차분한 안정형", "emoji": "🧘", "color": "#06b6d4"},
        {"id": "humorous", "type": "유머러스한 재미형", "emoji": "😄", "color": "#10b981"},
        {"id": "artist", "type": "창의적인 아티스트형", "emoji": "🎨", "color": "#f97316"},
        {"id": "analytical", "type": "지적이고 분석적인 형", "emoji": "🤓", "color": "#6366f1"},
        {"id": "active", "type": "활동적인 스포츠형", "emoji": "🏃", "color": "#ef4444"},
    ],
    "compatibilityJobs": ["라디오 DJ", "성우", "가수", "아나운서", "팟캐스터", "유튜버", "교사", "상담사", "배우", "MC", "오디오북 내레이터", "통역사", "콜센터 상담원", "강사"],
    "specialTags": ["ASMR 천재", "목소리 마약", "귀호강 주인공", "보이스 피셔", "음성 치료사", "힐링 보이스", "매력 발산기", "카리스마 폭발", "목소리 꿀", "보컬 DNA", "음성 마술사", "귀감 제조기"],
    "voiceColors": ["루비 레드", "사파이어 블루", "에메랄드 그린", "골든 옐로우", "아메시스트 퍼플", "다이아몬드 화이트", "오닉스 블랙", "로즈 골드", "실버 화이트", "코발트 블루"]
}

# --- 유틸 & 전처리 ---

def trim_and_normalize(y, sr):
    y = librosa.util.normalize(y)
    y, _ = librosa.effects.trim(y, top_db=30)
    return y

def median_filter_1d(x, k=5):
    if len(x) < k:
        return x
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode='edge')
    return np.array([np.median(xp[i:i + k]) for i in range(len(x))])

# --- 피치 통합 추출 (median Hz, std Hz, std cents) ---

def extract_pitch_stats(y, sr):
    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        frame_length=1024,
        hop_length=256,
        center=True
    )
    v = f0[~np.isnan(f0)]
    if len(v) == 0:
        return 150.0, 15.0, 80.0  # 안전 기본값
    v = median_filter_1d(v, k=5)
    med = float(np.median(v))
    if len(v) == 1:
        return med, 15.0, 80.0
    std_hz = float(np.std(v))
    std_cents = float(1200.0 * np.std(np.log2(v / med)))  # 피치 불변성
    return med, std_hz, std_cents

# --- 에너지 (레벨 무감화) ---

def analyze_energy(y):
    rms = librosa.feature.rms(y=y)[0]
    if np.all(rms == 0):
        return 0.0
    db = librosa.amplitude_to_db(rms, ref=np.max)  # 0..-inf
    # 평균 dB를 0..1로 매핑(약 -60~0dB 가정)
    norm = np.clip((db.mean() + 60.0) / 60.0, 0.0, 1.0)
    return float(norm)

# --- 말하기 속도 프록시 (ZCR + Spectral Flux 하이브리드) ---

def analyze_speaking_rate(y, sr):
    intervals = librosa.effects.split(y, top_db=30)
    yv = np.concatenate([y[s:e] for s, e in intervals]) if len(intervals) else y
    z = float(np.mean(librosa.feature.zero_crossing_rate(y=yv, frame_length=1024, hop_length=256)))
    S = np.abs(librosa.stft(yv, n_fft=1024, hop_length=256))
    flux = float(np.mean(np.maximum(0, np.diff(S, axis=1)).mean(axis=0)))
    proxy = 0.85 * z + 0.15 * (flux / np.maximum(S.mean(), 1e-6))  # 가벼운 보정
    # 경험적 범위: 0.03..0.18 → 70..180
    return float(70 + (np.clip(proxy, 0.03, 0.18) - 0.03) / (0.18 - 0.03) * (180 - 70))

# --- 조화도(HNR) 안전화 ---

def analyze_harmonicity(y):
    y_h, y_p = librosa.effects.hpss(y)
    he = float(np.sum(y_h**2) + 1e-12)
    pe = float(np.sum(y_p**2) + 1e-12)
    hnr = 10.0 * np.log10(he / pe)
    return float(np.clip(hnr, -20.0, 20.0))

# --- 스펙트럼 센트로이드 ---

def analyze_spectral_centroid(y, sr):
    return float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

# --- 나이대 추정 규칙 ---

def get_age_key_female(pitch, spectral_centroid, energy_norm01):
    """여성 목소리 나이대 추정 (에너지: 0~1 정규화 기준)"""
    if pitch > 245: return "10s"
    elif pitch > 220: return "20s_early"
    elif pitch > 200: return "20s_late"
    elif pitch > 185: return "30s_early" if energy_norm01 > 0.5 else "30s_late"
    elif pitch > 170: return "40s_early" if spectral_centroid > 2200 else "40s_late"
    else: return "50s_plus"

def get_age_key_male(pitch, spectral_centroid, energy_norm01):
    """남성 목소리 나이대 추정 (에너지: 0~1 정규화 기준)"""
    if pitch > 165: return "10s"
    elif pitch > 140: return "20s_early"
    elif pitch > 125: return "20s_late"
    elif pitch > 110: return "30s_early" if energy_norm01 > 0.5 else "30s_late"
    elif pitch > 95: return "40s_early" if spectral_centroid > 1800 else "40s_late"
    else: return "50s_plus"

# --- 프로필 산출 (치우침 완화 임계값 상향) ---

def get_voice_profile(features):
    profile = set()
    # pitch band
    if features['pitch'] > 180: profile.add('high_pitch')
    elif features['pitch'] < 130: profile.add('low_pitch')
    else: profile.add('mid_pitch')
    # energy (0~1)  ← high>0.75, low<0.30
    if features['energy'] > 0.75: profile.add('high_energy')
    elif features['energy'] < 0.30: profile.add('low_energy')
    else: profile.add('mid_energy')
    # clarity
    if features['harmonicity'] > 7.0: profile.add('clear_voice')
    elif features['harmonicity'] < 3.0: profile.add('husky_voice')
    else: profile.add('soft_voice')
    # stability: pitch std in cents ← dynamic>110c
    pitch_var = features.get('pitch_std_cents', None)
    if pitch_var is not None:
        profile.add('dynamic_tone' if pitch_var > 110.0 else 'stable_tone')
    else:
        profile.add('dynamic_tone' if features['pitch_std'] > 35 else 'stable_tone')
    return profile

# --- 직업 가중치 스코어러 (쏠림 방지) ---

def select_job_with_scores(profile, features, audio_bytes=None):
    jobs = [
        "아나운서","MC","오디오북 내레이터","배우","유튜버","상담사",
        "성우","가수","교사","팟캐스터","강사","통역사","라디오 DJ"
    ]
    scores = {j: 0.0 for j in jobs}

    tempo = float(features.get("tempo", 110.0))
    fast = tempo >= 120
    slow = tempo <= 95
    mid_speed = 95 < tempo < 130

    def add(job, w): scores[job] += w

    # 아나운서
    if 'clear_voice' in profile: add("아나운서", 2.5)
    if 'stable_tone' in profile: add("아나운서", 2.0)
    if 'mid_energy' in profile:  add("아나운서", 1.0)
    if mid_speed:                add("아나운서", 1.0)

    # MC (엄격: 빠른 속도 + 에너지 + 다이내믹)
    if 'high_energy' in profile and 'dynamic_tone' in profile: add("MC", 2.0)
    if fast:                                              add("MC", 2.0)
    if 'clear_voice' in profile:                          add("MC", 1.0)

    # 오디오북 내레이터
    if 'low_energy' in profile: add("오디오북 내레이터", 2.0)
    if 'soft_voice' in profile: add("오디오북 내레이터", 2.0)
    if 'stable_tone' in profile: add("오디오북 내레이터", 1.0)
    if slow:                     add("오디오북 내레이터", 1.5)

    # 배우
    if 'dynamic_tone' in profile: add("배우", 2.0)
    if 'husky_voice' in profile:  add("배우", 1.0)
    if 'high_energy' in profile:  add("배우", 0.5)

    # 유튜버
    if 'high_energy' in profile: add("유튜버", 1.5)
    if 'clear_voice' in profile: add("유튜버", 1.0)
    if 'dynamic_tone' in profile: add("유튜버", 1.0)

    # 상담사
    if 'soft_voice' in profile:  add("상담사", 2.0)
    if 'stable_tone' in profile: add("상담사", 1.0)
    if 'low_energy' in profile:  add("상담사", 1.0)

    # 성우
    if 'clear_voice' in profile: add("성우", 2.0)
    if 'dynamic_tone' in profile: add("성우", 1.0)
    if 'mid_energy' in profile:  add("성우", 1.0)

    # 가수
    if 'clear_voice' in profile: add("가수", 1.0)
    if 'high_pitch' in profile:  add("가수", 1.0)
    if 'dynamic_tone' in profile: add("가수", 0.5)

    # 교사
    if 'stable_tone' in profile: add("교사", 2.0)
    if 'mid_energy' in profile:  add("교사", 1.0)

    # 팟캐스터
    if 'dynamic_tone' in profile: add("팟캐스터", 1.0)
    if 'soft_voice' in profile:   add("팟캐스터", 1.0)
    if 'mid_energy' in profile:   add("팟캐스터", 0.5)

    # 강사
    if 'high_energy' in profile: add("강사", 2.0)
    if fast:                     add("강사", 1.0)
    if 'clear_voice' in profile: add("강사", 0.5)

    # 통역사
    if 'clear_voice' in profile: add("통역사", 1.0)
    if 'stable_tone' in profile: add("통역사", 1.0)
    if mid_speed:                add("통역사", 1.0)
    if 'low_energy' in profile:  add("통역사", 0.5)

    # 라디오 DJ
    if 'soft_voice' in profile:  add("라디오 DJ", 1.0)
    if 'low_energy' in profile:  add("라디오 DJ", 0.5)
    if 'stable_tone' in profile: add("라디오 DJ", 0.5)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top3 = ranked[:3]
    if audio_bytes is not None and len(top3) > 1 and top3[0][1] == top3[1][1]:
        h = int(hashlib.md5(audio_bytes).hexdigest(), 16)
        top3 = sorted(top3, key=lambda kv: (kv[1], (h ^ hash(kv[0])) & 0xffff), reverse=True)

    best_job = top3[0][0]
    candidates = [{"job": j, "score": round(s, 2)} for j, s in top3]
    return best_job, candidates

# --- 세부 결과 매핑 (직업=가중치 스코어러 사용) ---

def analyze_details_based_on_profile(profile, features=None, audio_bytes=None):
    """음성 프로필을 기반으로 모든 세부 결과를 결정"""
    voice_type_map = {
        ('high_energy', 'dynamic_tone'): "활기찬 에너지 보이스",
        ('high_pitch', 'clear_voice'): "맑고 청량한 크리스탈 보이스",
        ('low_pitch', 'stable_tone'): "깊고 카리스마 있는 베이스 보이스",
        ('low_energy', 'soft_voice'): "차분하고 속삭이는 위스퍼 보이스",
        ('clear_voice',): "따뜻하고 부드러운 허니 보이스",
        ('husky_voice', 'high_energy'): "파워풀하고 강렬한 썬더 보이스",
        ('mid_energy', 'dynamic_tone'): "감성적이고 몽환적인 문라이트 보이스",
    }
    voice_type = "시원하고 깔끔한 민트 보이스"
    for p_set, v_type in voice_type_map.items():
        if all(p in profile for p in p_set):
            voice_type = v_type
            break

    animal_type_map = {
        ('high_pitch', 'dynamic_tone'): "토끼상",
        ('high_pitch', 'low_energy'): "고양이상",
        ('high_pitch', 'high_energy'): "강아지상",
        ('low_pitch', 'husky_voice'): "늑대상",
        ('low_pitch', 'high_energy'): "사자상",
        ('mid_pitch', 'stable_tone'): "곰상",
        ('mid_pitch', 'dynamic_tone'): "여우상"
    }
    animal = "햄스터상"
    for p_set, a_type in animal_type_map.items():
        if all(p in profile for p in p_set):
            animal = a_type
            break
    animal_type = next(
        (item for item in voice_analysis_data["animalTypes"] if item["type"] == animal),
        random.choice(voice_analysis_data["animalTypes"])
    )

    personality_type_map = {
        ('high_energy', 'dynamic_tone'): "분위기 메이커형",
        ('low_energy', 'stable_tone'): "깊고 차분한 안정형",
        ('low_pitch', 'stable_tone'): "타고난 리더형",
        ('high_pitch', 'dynamic_tone'): "창의적인 아티스트형",
        ('high_energy', 'clear_voice'): "활동적인 스포츠형",
        ('husky_voice', 'dynamic_tone'): "유머러스한 재미형",
        ('clear_voice', 'stable_tone'): "지적이고 분석적인 형",
    }
    personality = "따뜻한 감성형"
    for p_set, p_type in personality_type_map.items():
        if all(p in profile for p in p_set):
            personality = p_type
            break
    personality_type = next(
        (item for item in voice_analysis_data["personalityTypes"] if item["type"] == personality),
        random.choice(voice_analysis_data["personalityTypes"])
    )

    # ▶ 가중치 스코어 기반 직업 선정
    if features is not None:
        job, job_candidates = select_job_with_scores(profile, features, audio_bytes)
    else:
        job = "라디오 DJ"
        job_candidates = [{"job": "라디오 DJ", "score": 0.0}]

    tag_map = {
        ('low_energy', 'soft_voice'): "ASMR 천재",
        ('clear_voice', 'high_pitch'): "귀호강 주인공",
        ('low_pitch', 'stable_tone'): "카리스마 폭발",
        ('soft_voice', 'clear_voice'): "목소리 꿀",
        ('high_energy', 'dynamic_tone'): "매력 발산기",
        ('husky_voice', 'low_pitch'): "보이스 피셔",
    }
    tag = "힐링 보이스"
    for p_set, t_type in tag_map.items():
        if all(p in profile for p in p_set):
            tag = t_type
            break

    color_map = {
        ('high_pitch', 'clear_voice'): "사파이어 블루",
        ('high_pitch', 'high_energy'): "골든 옐로우",
        ('low_pitch', 'stable_tone'): "오닉스 블랙",
        ('low_pitch', 'husky_voice'): "루비 레드",
        ('soft_voice', 'mid_pitch'): "로즈 골드",
        ('clear_voice', 'stable_tone'): "에메랄드 그린",
        ('dynamic_tone', 'mid_energy'): "아메시스트 퍼플",
    }
    color = "실버 화이트"
    for p_set, c_type in color_map.items():
        if all(p in profile for p in p_set):
            color = c_type
            break

    return {
        "voice_type": voice_type,
        "animal_type": {k: v for k, v in animal_type.items() if k not in ['id']},
        "personality_type": {k: v for k, v in personality_type.items() if k not in ['id']},
        "compatibility_job": job,
        "job_candidates": job_candidates,   # 상위 3개 후보 제공
        "special_tag": tag,
        "voice_color": color,
    }

# --- 레이더 정규화 (NaN 안전) ---

def normalize_features_for_radar(features):
    # pitch: 80~250Hz → 0~100
    pitch_score = np.nan_to_num((features["pitch"] - 80) / (250 - 80) * 100, nan=50.0)
    # energy: 0~1 → 0~100
    energy_score = np.nan_to_num(features["energy"] * 100, nan=50.0)
    # speaking rate(tempo 자리): 70~180 → 0~100
    tempo_score = np.nan_to_num((features["tempo"] - 70) / (180 - 70) * 100, nan=50.0)
    # HNR → 완만 스케일 (2.5*dB + 50)
    clearness_score = np.nan_to_num(features['harmonicity'] * 2.5 + 50, nan=50.0)
    # 안정감: 센트 표준편차 사용 시 더 타당 (작을수록 안정)
    if "pitch_std_cents" in features:
        c = float(features["pitch_std_cents"])
        # 40→90, 120→50, 200→10 로 매핑(구간 외는 양끝으로 포화)
        stability_score = np.interp(c, [40, 120, 200], [90, 50, 10])
    else:
        # Hz 표준편차일 때도 대략적 보정(피치가 높을수록 유리하므로 보수적으로)
        h = float(features["pitch_std"])
        stability_score = np.interp(h, [5, 20, 40], [90, 50, 10])

    stability_score = float(np.clip(stability_score, 10, 100))

    return [
        {"feature": "높이", "value": int(np.clip(pitch_score, 10, 100))},
        {"feature": "에너지", "value": int(np.clip(energy_score, 10, 100))},
        {"feature": "속도", "value": int(np.clip(tempo_score, 10, 100))},
        {"feature": "맑음", "value": int(np.clip(clearness_score, 10, 100))},
        {"feature": "안정감", "value": int(np.clip(stability_score, 10, 100))},
    ]

def calculate_attraction_score(radar_data):
    values = [item['value'] for item in radar_data]
    average_score = np.mean(values)
    std_dev = np.std(values)
    balance_bonus = max(0, 25 - std_dev)
    ideal_range_bonus = 10 if all(40 <= v <= 85 for v in values) else 0
    final_score = 60 + (average_score - 50) * 0.5 + balance_bonus + ideal_range_bonus
    return min(99, max(60, int(final_score)))

# --- 품질 진단 ---

def estimate_snr(y):
    rms = librosa.feature.rms(y=y)[0]
    med = np.median(rms)
    noise = np.percentile(rms, 10)
    return float(20 * np.log10((med + 1e-8) / (noise + 1e-8)))

def clipping_ratio(y):
    return float(np.mean(np.abs(y) > 0.98))

def choose_deterministic(seq, audio_bytes):
    if not seq:
        return None
    h = int(hashlib.md5(audio_bytes).hexdigest(), 16)
    return seq[h % len(seq)]

# --- 엔드포인트 ---

@app.post("/api/analyze")
async def analyze_voice(gender: str = Form(...), audio: UploadFile = File(...)):
    try:
        # content-type 가드(브라우저 업로드는 종종 application/octet-stream)
        if audio.content_type and not (
            audio.content_type.startswith("audio/") or audio.content_type == "application/octet-stream"
        ):
            raise HTTPException(status_code=400, detail=f"지원하지 않는 콘텐츠 타입입니다: {audio.content_type}")

        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="오디오 파일이 비어 있습니다.")
        if len(audio_bytes) > MAX_BYTES:
            raise HTTPException(status_code=400, detail=f"파일이 너무 큽니다(≤ {MAX_BYTES // (1024*1024)}MB).")

        # pydub 로드
        try:
            sound = AudioSegment.from_file(io.BytesIO(audio_bytes))
        except Exception:
            raise HTTPException(status_code=400, detail="오디오 포맷을 인식할 수 없습니다. WAV/MP3 등 표준 포맷을 사용해 주세요.")

        # 모노 변환
        if sound.channels > 1:
            sound = sound.set_channels(1)

        # float32 파형
        samples = np.array(sound.get_array_of_samples()).astype(np.float32) / (2 ** (sound.sample_width * 8 - 1))

        # 리샘플
        target_sr = 22050
        y = librosa.resample(y=samples, orig_sr=sound.frame_rate, target_sr=target_sr)
        sr = target_sr

        # 전처리
        y = trim_and_normalize(y, sr)
        if len(y) < int(MIN_SEC * sr):
            raise HTTPException(status_code=400, detail=f"오디오 길이가 너무 짧습니다(≥{MIN_SEC:.1f}초 필요).")
        if len(y) > int(MAX_SEC * sr):
            y = y[:int(MAX_SEC * sr)]

        # 특성 추출 (pyin 1회)
        pitch_med, pitch_std_hz, pitch_std_cents = extract_pitch_stats(y, sr)
        energy = analyze_energy(y)                    # 0~1
        speaking_rate = analyze_speaking_rate(y, sr)  # 70~180
        hnr = analyze_harmonicity(y)                  # -20~20 (클리핑)
        spec_cent = analyze_spectral_centroid(y, sr)

        features = {
            "pitch": pitch_med,
            "energy": energy,
            "tempo": speaking_rate,              # 기존 키 유지(UI 호환)
            "pitch_std": pitch_std_hz,
            "pitch_std_cents": pitch_std_cents,
            "harmonicity": hnr
        }

        voice_profile = get_voice_profile(features)
        detailed_results = analyze_details_based_on_profile(voice_profile, features, audio_bytes)

        # 나이대 추정
        if gender == 'female':
            age_key = get_age_key_female(features["pitch"], spec_cent, features["energy"])
        else:  # male or other
            age_key = get_age_key_male(features["pitch"], spec_cent, features["energy"])

        age_info = voice_analysis_data["age_groups"].get(age_key, voice_analysis_data["age_groups"]["30s_early"])

        # 레이더/스코어
        radar_data = normalize_features_for_radar(features)
        attraction_score = calculate_attraction_score(radar_data)

        # 유니크니스(안전 계산)
        uniqueness_score = int(np.clip(
            65
            + 0.20 * np.nan_to_num(features.get("pitch_std_cents", 80.0))
            + 25.0 * abs(np.nan_to_num(features["energy"]) - 0.5)
            + 1.5 * abs(np.nan_to_num(features["harmonicity"]) - 5.0),
            60, 99
        ))

        # 입력 품질 경고
        snr = estimate_snr(y)
        clip = clipping_ratio(y)
        voiced_ratio = float(sum((e - s) for s, e in librosa.effects.split(y, top_db=30)) / len(y))
        warnings = []
        if snr < 10:
            warnings.append("주변 소음이 커서 정확도가 떨어질 수 있어요.")
        if clip > 0.02:
            warnings.append("입력이 클리핑되었습니다. 마이크 입력 레벨을 낮춰주세요.")
        if voiced_ratio < 0.4:
            warnings.append("무음 구간이 많습니다. 2초 이상 또박또박 말해주세요.")

        # 결정적 유머 선택(같은 파일 → 같은 문구)
        humor = choose_deterministic(age_info["humor"], audio_bytes)

        result = {
            "version": API_VERSION,
            "age_range": age_info["range"],
            "humor_quote": humor,
            "attraction_score": attraction_score,
            "uniqueness_score": uniqueness_score,
            "radar_data": radar_data,
            "warnings": warnings,
            **detailed_results
        }

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred during analysis: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Voice Age API is running!", "version": API_VERSION}

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "version": API_VERSION}
