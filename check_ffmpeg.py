# check_ffmpeg.py

import shutil
import os

print("--- FFmpeg 찾기 테스트 시작 ---")

# 1. shutil.which를 사용해 FFmpeg 실행 파일 찾기
ffmpeg_path = shutil.which('ffmpeg')

if ffmpeg_path:
    print(f"성공! FFmpeg를 찾았습니다.")
    print(f"위치: {ffmpeg_path}")
else:
    print("실패: 파이썬이 시스템 경로(PATH)에서 FFmpeg를 찾을 수 없습니다.")
    print("\n--- 시스템 경로(PATH) 변수 내용 ---")
    path_variable = os.environ.get('PATH', 'PATH 변수를 찾을 수 없음')
    # 경로를 한 줄씩 보기 좋게 출력
    for p in path_variable.split(';'):
        print(p)
    print("---------------------------------")
    print("\n해결책: 위 경로 목록에 FFmpeg가 설치된 폴더(예: C:\\ProgramData\\chocolatey\\bin)가 없는 것이 원인입니다.")

print("--- 테스트 종료 ---")