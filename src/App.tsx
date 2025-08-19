// src/App.tsx

import React, { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import {
  Mic, MicOff, Play, Square, Users, Sparkles, RefreshCw,
  AlertCircle, X, Share2, Download, Heart, Star, Trophy, Zap, Music, ArrowRight, ArrowLeft, Info, CheckCircle2
} from 'lucide-react';
import html2canvas from 'html2canvas';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts';
import './App.css';

// --- 상수 데이터 ---
const sampleTexts: string[] = [
  "안녕하세요! 저는 커피를 정말 좋아하고, 주말에는 드라마 정주행을 즐겨요.",
  "오늘 날씨가 정말 좋네요. 이런 날엔 산책하면서 음악 듣는 게 최고인 것 같아요.",
  "제가 가장 좋아하는 음식은 치킨인데요, 친구들과 함께 먹으면 더 맛있어요.",
  "책 읽기와 영화 보기를 좋아해요. 특히 감동적인 스토리에 약한 편이에요.",
  "여행을 정말 좋아해서 새로운 곳을 탐험하는 게 제 취미 중 하나예요.",
];

// --- 타입 정의 ---
type Status = 'idle' | 'recording' | 'recorded' | 'analyzing' | 'result';
interface AnalysisResult {
  age_range: string;
  humor_quote: string;
  voice_type: string;
  attraction_score: number;
  animal_type: { type: string; emoji: string; desc: string };
  personality_type: { type: string; emoji: string; color: string };
  compatibility_job: string;
  special_tag: string;
  voice_color: string;
  uniqueness_score: number;
  radar_data: { feature: string; value: number }[];
}
interface RecordedAudio { blob: Blob; url: string; }

// --- 헬퍼 함수 ---
const getRandomSample = () => sampleTexts[Math.floor(Math.random() * sampleTexts.length)];

// 원형 진행 표시(10초)
const RecordProgress: React.FC<{ sec: number; max?: number }> = ({ sec, max = 10 }) => {
  const pct = Math.min(100, (sec / max) * 100);
  const radius = 54, stroke = 8, C = 2 * Math.PI * radius;
  const dash = (pct / 100) * C;
  return (
    <svg width="130" height="130" role="progressbar" aria-valuenow={Math.floor(pct)} aria-valuemin={0} aria-valuemax={100}>
      <circle cx="65" cy="65" r={radius} stroke="rgba(255,255,255,.25)" strokeWidth={stroke} fill="none" />
      <circle cx="65" cy="65" r={radius} stroke="#22c55e" strokeWidth={stroke} fill="none"
        strokeDasharray={`${dash} ${C - dash}`} strokeLinecap="round" transform="rotate(-90 65 65)" />
      <text x="50%" y="50%" textAnchor="middle" dominantBaseline="central" fontWeight={800} fontSize="20" fill="#fff">
        {sec.toFixed(1)}s
      </text>
    </svg>
  );
};

// 레이더 차트
const VoiceRadarChart: React.FC<{ data: { feature: string; value: number }[] }> = ({ data }) => (
  <div className="radar-chart-container">
    <ResponsiveContainer width="100%" height={250}>
      <RadarChart cx="50%" cy="50%" outerRadius="80%" data={data}>
        <PolarGrid stroke="rgba(255, 255, 255, 0.5)" />
        <PolarAngleAxis dataKey="feature" tick={{ fill: 'white', fontSize: 13, fontWeight: 600 }} />
        <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
        <Radar name="Voice" dataKey="value" stroke="#fbbf24" fill="#fbbf24" fillOpacity={0.6} />
      </RadarChart>
    </ResponsiveContainer>
  </div>
);

const ResultCaptureCard: React.FC<{ result: AnalysisResult }> = ({ result }) => (
  <div id="resultCardToCapture" className="share-card">
    <div className="share-card__frame" />
    <div className="share-card__ribbon">
      <div className="ribbon__logo"><Sparkles size={18} /></div>
      <span className="ribbon__brand">VOICEAGE</span>
      <span className="ribbon__tagline">AI Voice Personality</span>
    </div>
    <div className="share-card__content">
      <div className="share-card__title">
        <h2>목소리 분석 결과</h2>
        <span className="badge">{result.voice_color}</span>
      </div>
      <div className="share-card__headline">
        <div className="headline__age">
          <span className="age__label">목소리 나이</span>
          <span className="age__value">{result.age_range}</span>
        </div>
        <div className="headline__metrics">
          <div className="metric">
            <Heart size={16} /><span className="metric__label">매력도</span><span className="metric__value">{result.attraction_score}</span>
          </div>
          <div className="metric">
            <Star size={16} /><span className="metric__label">유니크</span><span className="metric__value">{result.uniqueness_score}%</span>
          </div>
        </div>
      </div>
      <blockquote className="share-card__quote">“{result.humor_quote}”</blockquote>
      <div className="share-card__grid">
        <div className="mini-card">
          <div className="mini-card__head"><span className="emoji">{result.animal_type.emoji}</span><span className="mini-card__title">동물상</span></div>
          <div className="mini-card__value">{result.animal_type.type}</div>
          <div className="mini-card__desc">{result.animal_type.desc}</div>
        </div>
        <div className="mini-card">
          <div className="mini-card__head"><span className="emoji" style={{ color: result.personality_type.color }}>{result.personality_type.emoji}</span><span className="mini-card__title">성격</span></div>
          <div className="mini-card__value">{result.personality_type.type}</div>
        </div>
        <div className="mini-card">
          <div className="mini-card__head"><Zap size={16} /><span className="mini-card__title">보이스</span></div>
          <div className="mini-card__value">{result.voice_type}</div>
        </div>
        <div className="mini-card">
          <div className="mini-card__head"><span className="emoji">💼</span><span className="mini-card__title">어울리는 직업</span></div>
          <div className="mini-card__value">{result.compatibility_job}</div>
        </div>
      </div>
      <div className="share-card__hashtags">
        <span className="tag">#{result.special_tag.replace(/\s/g, '')}</span>
        <span className="tag">#{result.voice_color.replace(/\s/g, '')}</span>
        <span className="tag">#VoiceAge</span>
      </div>
    </div>
    <div className="share-card__footer">
      <div className="footer__brand"><Sparkles size={14} /><span>VOICEAGE</span></div>
      <span className="footer__url">voiceage.app</span>
    </div>
  </div>
);

// 인사이트 칩 계산
const computeInsightChips = (radar: AnalysisResult['radar_data']) => {
  const labelPos: Record<string, string> = {'높이': '고음도 높음', '에너지': '에너지 넘침', '속도': '말 속도 빠름', '맑음': '맑은 톤', '안정감': '안정감 높음'};
  const labelNeg: Record<string, string> = {'높이': '저음 기조', '에너지': '차분함', '속도': '말 속도 느림', '맑음': '허스키 톤', '안정감': '톤 변동 큼'};
  const top = [...radar].sort((a, b) => b.value - a.value).slice(0, 3);
  return top.map(({ feature, value }) => value >= 60 ? (labelPos[feature] || feature) : value <= 40 ? (labelNeg[feature] || feature) : feature);
};

// 공유 문구
const shareText = (r: AnalysisResult) => `🎤 내 보이스 에이지: ${r.age_range}\n“${r.humor_quote}”\n#${r.special_tag.replace(/\s/g,'')} #${r.voice_color.replace(/\s/g,'')} #VoiceAge`;

const VoiceAgeApp: React.FC = () => {
  const [status, setStatus] = useState<Status>('idle');
  const [gender, setGender] = useState<'male' | 'female' | null>(null);
  const [recordedAudio, setRecordedAudio] = useState<RecordedAudio | null>(null);
  const [recordingTime, setRecordingTime] = useState<number>(0);
  const [totalUsers, setTotalUsers] = useState<number>(15247);
  const [currentSample, setCurrentSample] = useState<string>(getRandomSample());
  const [error, setError] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const [resultView, setResultView] = useState<'summary' | 'details'>('summary');
  const [toast, setToast] = useState<string | null>(null);
  const [showPrivacy, setShowPrivacy] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recordingTimeRef = useRef(recordingTime);
  const stopMeterRef = useRef<(() => void) | null>(null);
  const titleRef = useRef<HTMLHeadingElement | null>(null);

  useEffect(() => {
    const interval = setInterval(() => setTotalUsers(prev => prev + Math.floor(Math.random() * 5) + 1), 3000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    let timer: number | undefined;
    if (status === 'recording') {
      timer = window.setInterval(() => {
        setRecordingTime(prev => {
          if (prev >= 9.9) {
            if (mediaRecorderRef.current?.state === 'recording') mediaRecorderRef.current.stop();
            return 10;
          }
          return +(prev + 0.1).toFixed(1);
        });
      }, 100);
    }
    return () => { if (timer) window.clearInterval(timer); };
  }, [status]);

  useEffect(() => { recordingTimeRef.current = recordingTime; }, [recordingTime]);

  const setupLiveMeter = (stream: MediaStream) => {
    const Ctx = (window as any).AudioContext || (window as any).webkitAudioContext;
    if (!Ctx) return () => {};
    const ctx = new Ctx();
    const src = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser(); analyser.fftSize = 256;
    src.connect(analyser);
    const data = new Uint8Array(analyser.frequencyBinCount);
    const bars = 16;
    let raf = 0;
    const tick = () => {
      analyser.getByteTimeDomainData(data);
      let sum = 0;
      for (let i = 0; i < data.length; i++) {
        const v = (data[i] - 128) / 128; sum += v * v;
      }
      const rms = Math.sqrt(sum / data.length);
      const arr = Array.from({ length: bars }, (_, i) => Math.max(0, Math.min(1, rms * (1 + i * 0.15))));
      const root = document.documentElement;
      root.style.setProperty('--meter-rms', String(rms));
      root.style.setProperty('--meter-bars', String(bars));
      arr.forEach((lv, idx) => root.style.setProperty(`--meter-${idx}`, String(lv)));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => { cancelAnimationFrame(raf); try { ctx.close(); } catch {} };
  };

  const showToast = (t: string) => { setToast(t); window.setTimeout(() => setToast(null), 2200); };

  const startRecording = useCallback(async () => {
    setError(null); if (status !== 'idle' || !gender) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 44100, channelCount: 1, echoCancellation: true, noiseSuppression: true } });
      streamRef.current = stream; const audioChunks: Blob[] = [];
      const mimeType = 'audio/webm;codecs=opus';
      const mediaRecorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = mediaRecorder;
      stopMeterRef.current = setupLiveMeter(stream);
      mediaRecorder.ondataavailable = (event: BlobEvent) => { if (event.data.size > 0) audioChunks.push(event.data); };
      mediaRecorder.onstop = () => {
        stopMeterRef.current?.(); stopMeterRef.current = null;
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        if (recordingTimeRef.current < 2) {
          setError("너무 짧아요! 2초 이상 녹음해야 정확한 분석이 가능해요.");
          setCurrentSample(getRandomSample());
          setStatus('idle'); setRecordingTime(0); setRecordedAudio(null);
        } else {
          const audioUrl = URL.createObjectURL(audioBlob);
          setRecordedAudio({ blob: audioBlob, url: audioUrl });
          setStatus('recorded');
        }
        if (streamRef.current) streamRef.current.getTracks().forEach(track => track.stop());
      };
      mediaRecorder.start(); setStatus('recording'); setRecordingTime(0);
    } catch (err) {
      console.error('마이크 접근 실패:', err);
      setError('마이크 권한이 필요해요. 브라우저 설정을 확인해주세요!');
    }
  }, [status, gender]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') mediaRecorderRef.current.stop();
  }, []);

  const playRecording = useCallback(() => { if (recordedAudio) new Audio(recordedAudio.url).play(); }, [recordedAudio]);

  const getNewSample = () => {
    let newSample;
    do { newSample = getRandomSample(); } while (newSample === currentSample);
    setCurrentSample(newSample);
  };

  const resetAll = useCallback(() => {
    if (recordedAudio) URL.revokeObjectURL(recordedAudio.url);
    setRecordedAudio(null); setRecordingTime(0); setStatus('idle'); setError(null); setAnalysisResult(null); setResultView('summary'); setGender(null);
  }, [recordedAudio]);

  const analyzeVoice = useCallback(async () => {
    if (!recordedAudio || !gender) { setError("성별을 선택해야 분석을 시작할 수 있습니다."); return; }
    setStatus('analyzing'); setError(null);
    const formData = new FormData();
    formData.append('gender', gender);
    formData.append('audio', recordedAudio.blob, 'recording.webm');
    try {
      const apiUrl = `${import.meta.env.VITE_API_URL}/api/analyze`;
      const response = await fetch(apiUrl, { method: 'POST', body: formData });
      // const response = await fetch('/api/analyze', { method: 'POST', body: formData });
      if (!response.ok) {
        let msg = '분석 중 에러가 발생했습니다.';
        try { const errorData = await response.json(); msg = errorData.detail || msg; } catch {}
        throw new Error(msg);
      }
      const result: AnalysisResult = await response.json();
      setAnalysisResult(result); setStatus('result');
      setTimeout(() => titleRef.current?.focus(), 30);
    } catch (err: any) {
      console.error('분석 요청 실패:', err);
      setError(err.message || '분석 서버에 연결할 수 없습니다.');
      showToast('연결이 불안정해요. 3초 후 자동 재시도합니다.');
      setStatus('recorded');
      setTimeout(() => analyzeVoice(), 3000);
    }
  }, [recordedAudio, gender]);

  const captureResultCard = useCallback(async (): Promise<Blob | null> => {
    setIsCapturing(true);
    await new Promise(resolve => setTimeout(resolve, 100));
    const cardElement = document.getElementById('resultCardToCapture') || document.querySelector('.result-container');
    if (!cardElement) { console.error('캡처할 결과 카드 요소를 찾을 수 없습니다.'); setIsCapturing(false); return null; }
    try {
      const scale = Math.min(3, Math.max(2, (window.devicePixelRatio || 1) * 1.25));
      const canvas = await html2canvas(cardElement as HTMLElement, { backgroundColor: '#6B5FAA', useCORS: true, scale });
      return new Promise(resolve => canvas.toBlob(blob => resolve(blob), 'image/png'));
    } catch (error) {
      console.error('결과 카드 이미지 변환 실패:', error); return null;
    } finally { setIsCapturing(false); }
  }, []);

  const shareResult = useCallback(async () => {
    if (!analysisResult) return;
    const blob = await captureResultCard(); if (!blob) { alert('이미지를 생성하는 데 실패했습니다.'); return; }
    const file = new File([blob], 'voice-age-result.png', { type: 'image/png' });
    const text = shareText(analysisResult);
    const shareData: any = { title: '보이스에이지 - 내 목소리 분석 결과!', text, url: window.location.href };
    if (navigator.canShare?.({ files: [file] })) shareData.files = [file];
    try {
      if (navigator.share) await navigator.share(shareData);
      else throw new Error('web share not supported');
    } catch {
      try {
        await navigator.clipboard.writeText(`${text}\n${window.location.href}`);
        alert('공유 문구와 링크를 복사했어요! 채팅창에 붙여넣어 공유해보세요.');
      } catch (e) {
        console.error('클립보드 복사 실패:', e);
        alert('복사에 실패했습니다. "저장하기"를 이용해주세요.');
      }
    }
  }, [analysisResult, captureResultCard]);

  const downloadResult = useCallback(async () => {
    const blob = await captureResultCard(); if (!blob) { alert('이미지를 생성하는 데 실패했습니다.'); return; }
    const url = URL.createObjectURL(blob); const link = document.createElement('a');
    link.href = url; link.download = '보이스에이지-결과.png'; document.body.appendChild(link);
    link.click(); document.body.removeChild(link); URL.revokeObjectURL(url);
  }, [captureResultCard]);

  const insightChips = useMemo(() => analysisResult ? computeInsightChips(analysisResult.radar_data) : [], [analysisResult]);

  const renderMainContent = () => {
    switch (status) {
      case 'recording': return (<div className="status-content"><RecordProgress sec={recordingTime} /><div className="recording-text"><div className="info">녹음 중...</div><div className="recording-wave" aria-hidden="true">{Array.from({ length: 16 }).map((_, i) => (<div key={i} className="wave-bar" style={{ height: `calc(8px + var(--meter-${i}, 0) * 32px)` }} />))}</div></div></div>);
      case 'recorded': return (<div className="status-content"><div className="recorded-icon"><Square size={48} color="white" /></div><div className="recorded-text"><h2>녹음 완료!</h2><p aria-live="polite">{recordingTime.toFixed(1)}초 분량</p><div className="recorded-preview"><Music size={16} /><span>목소리 데이터 준비됨</span></div></div></div>);
      case 'analyzing': return (<div className="status-content"><div className="analyzing-icon"><div className="analyzing-icon-inner"><Sparkles size={32} color="white" /></div></div><div className="analyzing-text"><h2>AI 분석 중...</h2><p>목소리 DNA를 확인하고 있어요! (약 2~3초)</p><div className="analyzing-steps"><div className="step active">음성 파형 분석</div><div className="step active">톤 분석</div><div className="step loading">성격 유형 매칭</div></div></div></div>);
      case 'result':
        if (!analysisResult) return null;
        if (resultView === 'summary') {
          const isUnique = analysisResult.uniqueness_score >= 90;
          return (<div className="result-container result-summary"><div className="result-header"><h2 ref={titleRef} tabIndex={-1} className="result-title" aria-live="polite">{analysisResult.age_range}의 목소리 {isUnique && <span className="badge-unique">UNIQUE 90+</span>}</h2><p className="humor-quote">"{analysisResult.humor_quote}"</p></div>{insightChips.length > 0 && (<div className="insight-chips" aria-label="목소리 인사이트">{insightChips.map((chip, i) => <span key={i} className="chip">{chip}</span>)}</div>)}<h3 className="radar-title">목소리 특징 분석</h3><VoiceRadarChart data={analysisResult.radar_data} /><div className="scores-row"><div className="score-item"><Heart className="score-icon" size={20} /><span className="score-label">매력도</span><span className="score-value">{analysisResult.attraction_score}점</span><p className="score-desc">목소리의 안정감과 편안함</p></div><div className="score-item"><Star className="score-icon" size={20} /><span className="score-label">유니크</span><span className="score-value">{analysisResult.uniqueness_score}%</span><p className="score-desc">다른 목소리와의 차별성</p></div></div><button onClick={() => setResultView('details')} className="btn btn-details" aria-label="상세 결과 보기">자세한 결과 보기 <ArrowRight size={18} /></button></div>);
        } else {
          return (<div className="result-container result-details"><div className="result-header"><h2 className="result-title">상세 분석 결과</h2></div><div className="result-details-grid"><div className="detail-card animal-card"><div className="card-header"><span className="animal-emoji">{analysisResult.animal_type.emoji}</span><div><h4>목소리 동물상</h4><p>{analysisResult.animal_type.type}</p></div></div><span className="card-desc">{analysisResult.animal_type.desc}</span></div><div className="detail-card personality-card"><div className="card-header"><span className="personality-emoji" style={{ color: analysisResult.personality_type.color }}>{analysisResult.personality_type.emoji}</span><div><h4>성격 유형</h4><p>{analysisResult.personality_type.type}</p></div></div></div><div className="detail-card voice-card"><div className="card-header"><Zap size={24} color="#8b5cf6" /><div><h4>목소리 타입</h4><p>{analysisResult.voice_type}</p></div></div></div><div className="detail-card job-card"><div className="card-header"><span className="job-emoji">💼</span><div><h4>어울리는 직업</h4><p>{analysisResult.compatibility_job}</p></div></div></div></div><div className="special-tags"><span className="special-tag">#{analysisResult.special_tag}</span><span className="special-tag">#{analysisResult.voice_color}</span></div><button onClick={() => setResultView('summary')} className="btn btn-back" aria-label="요약으로 돌아가기"><ArrowLeft size={18} /> 요약으로 돌아가기</button></div>);
        }
      case 'idle':
      default:
        return (
          <div className="status-content">
            <div className="gender-selector">
              <h3 className="gender-title">성별을 선택해주세요</h3>
              <div className="gender-buttons">
                <button className={`btn-gender ${gender === 'male' ? 'selected' : ''}`} onClick={() => setGender('male')} aria-pressed={gender === 'male'}>남성 {gender === 'male' && <CheckCircle2 size={20} />}</button>
                <button className={`btn-gender ${gender === 'female' ? 'selected' : ''}`} onClick={() => setGender('female')} aria-pressed={gender === 'female'}>여성 {gender === 'female' && <CheckCircle2 size={20} />}</button>
              </div>
            </div>
            <button className="idle-icon" onClick={startRecording} disabled={!gender} aria-label="녹음 시작"><div className="mic-pulse"><Mic size={64} color="white" /></div></button>
            <div className="idle-text">
              <h2>{gender ? "목소리를 들려주세요!" : "성별 선택 후 시작"}</h2>
              <p>10초면 충분해요</p>
            </div>
          </div>
        );
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="title-group"><Sparkles size={32} className="sparkle-icon" /><h1>보이스에이지</h1><Sparkles size={32} className="sparkle-icon" /></div>
        <p className="subtitle">AI가 분석하는 내 목소리의 모든 것</p>
        <div className="header-row">
          <div className="user-counter"><Users size={20} /><span>지금까지 {totalUsers.toLocaleString()}명이 테스트했어요!</span></div>
          <button className="privacy-btn" aria-label="프라이버시 안내" onClick={() => setShowPrivacy(true)}><Info size={18} /> 정보</button>
        </div>
      </header>
      <main className="main-content">
        <div className="card">
          <div className="card-status-display">{renderMainContent()}</div>
          {(status === 'idle' || status === 'recording') && (<div className="sample-text-box" aria-live="polite"><div className="sample-text-header"><span>💬 이런 식으로 말해보세요:</span><button onClick={getNewSample} title="다른 예시 보기" aria-label="샘플 문장 바꾸기"><RefreshCw size={16} /></button></div><p>"{currentSample}"</p></div>)}
          {error && (<div className="error-box" role="alert"><AlertCircle size={20} /><span>{error}</span><button onClick={() => setError(null)} aria-label="오류 메시지 닫기"><X size={20} /></button></div>)}
          <div className="button-group">
            {status === 'idle' && (<button onClick={startRecording} className="btn btn-primary" disabled={!gender} aria-label="녹음 시작"><Mic size={20} />{gender ? '녹음 시작' : '성별을 먼저 선택하세요'}</button>)}
            {status === 'recording' && (<button onClick={stopRecording} className="btn btn-danger" aria-label="녹음 중지"><MicOff size={20} />녹음 중지</button>)}
            {status === 'recorded' && (<><button onClick={playRecording} className="btn btn-secondary" aria-label="녹음 재생"><Play size={18} />녹음 재생</button><div className="grid-buttons"><button onClick={resetAll} className="btn btn-light" aria-label="다시 녹음">다시 녹음</button><button onClick={analyzeVoice} className="btn btn-accent" aria-label="분석 시작"><Sparkles size={18} />분석 시작!</button></div></>)}
            {status === 'result' && (<button onClick={resetAll} className="btn btn-primary" aria-label="다시 분석하기"><RefreshCw size={20} /> 다시 분석하기</button>)}
          </div>
        </div>
      </main>
      <footer className="app-footer"><p>🎯 목소리 나이대 + 성격 + 매력도 + 동물상까지!</p></footer>
      {status === 'result' && analysisResult && (<div className="cta-sticky" role="region" aria-label="공유 및 저장"><button onClick={shareResult} className="btn btn-share" aria-label="결과 공유하기"><Share2 size={18} />공유</button><button onClick={downloadResult} className="btn btn-download" aria-label="결과 저장하기"><Download size={18} />저장</button></div>)}
      {isCapturing && analysisResult && (<div className="capture-container"><ResultCaptureCard result={analysisResult} /></div>)}
      {isCapturing && !analysisResult && (<div className="modal-overlay"><div className="share-modal"><h3>결과 이미지 생성 중...</h3><p>잠시만 기다려주세요!</p><div className="analyzing-icon"><div className="analyzing-icon-inner"><Download size={32} color="white" /></div></div></div></div>)}
      {showPrivacy && (<div className="modal-overlay" onClick={() => setShowPrivacy(false)}><div className="privacy-modal" onClick={(e) => e.stopPropagation()}><div className="privacy-head"><h3>프라이버시 안내</h3><button aria-label="닫기" onClick={() => setShowPrivacy(false)}><X size={18} /></button></div><ul className="privacy-list"><li>🎤 녹음은 분석 목적에만 사용되며 서버에 저장하지 않습니다.</li><li>⏱ 10초 이내의 짧은 발화가 가장 정확합니다.</li><li>🔇 조용한 환경에서 진행하면 정확도가 올라갑니다.</li></ul><button className="btn btn-primary" onClick={() => setShowPrivacy(false)}>확인</button></div></div>)}
      {toast && <div className="toast" role="status" aria-live="polite">{toast}</div>}
    </div>
  );
};

export default VoiceAgeApp;