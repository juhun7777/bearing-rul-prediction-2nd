"""
깔끔한 출력을 위한 유틸리티 함수들
"""

import sys
import time
from contextlib import contextmanager

@contextmanager
def quiet_loading(message="처리 중"):
    """조용한 로딩 컨텍스트 매니저"""
    print(f"   {message}...", end="", flush=True)
    start_time = time.time()
    
    try:
        yield
        elapsed = time.time() - start_time
        print(f" ✅ 완료 ({elapsed:.1f}초)")
    except Exception as e:
        print(f" ❌ 실패: {str(e)}")
        raise

@contextmanager
def suppress_prints():
    """print 출력을 임시로 억제"""
    original_stdout = sys.stdout
    sys.stdout = open('nul', 'w') if sys.platform == 'win32' else open('/dev/null', 'w')
    
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout

def print_progress_simple(current, total, prefix="진행"):
    """간단한 진행률 표시"""
    if current % max(1, total // 10) == 0 or current == total:
        percent = (current / total) * 100
        print(f"\r   {prefix}: {current}/{total} ({percent:.0f}%)", end="", flush=True)
        if current == total:
            print(" ✅")

class QuietProgress:
    """조용한 진행률 표시 클래스"""
    
    def __init__(self, total, description="처리"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
        print(f"   {description} 시작...")
    
    def update(self, n=1):
        self.current += n
        # 10% 단위로만 출력
        if self.current % max(1, self.total // 10) == 0 or self.current == self.total:
            percent = (self.current / self.total) * 100
            elapsed = time.time() - self.start_time
            if self.current == self.total:
                print(f"   ✅ {self.description} 완료 ({self.current}/{self.total}, {elapsed:.1f}초)")
            else:
                print(f"   📊 {self.description}: {percent:.0f}% ({self.current}/{self.total})")
    
    def close(self):
        if self.current < self.total:
            elapsed = time.time() - self.start_time
            print(f"   ✅ {self.description} 완료 ({self.current}/{self.total}, {elapsed:.1f}초)")
