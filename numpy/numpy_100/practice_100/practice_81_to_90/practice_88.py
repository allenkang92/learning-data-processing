# 88. How to implement the Game of Life using numpy arrays?

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

# Conway의 생명 게임 규칙:
# 1. 살아있는 셀의 이웃이 2-3개면 살아남고, 그렇지 않으면 죽음 (과밀 또는 과소)
# 2. 죽은 셀의 이웃이 정확히 3개면 살아남 (번식)

def game_of_life_step(grid):
    """생명 게임의 한 스텝을 수행하여 다음 세대의 격자를 반환합니다."""
    # 주변 8개 이웃의 합을 계산 (경계에서는 0으로 간주)
    # 1. 원본 배열을 복사하고 패딩 추가
    padded_grid = np.pad(grid, pad_width=1, mode='constant', constant_values=0)
    
    # 2. 모든 셀에 대해 주변 8개 이웃의 합 계산
    neighbors_sum = np.zeros_like(padded_grid)
    
    # 주변 8방향의 이웃을 더함
    neighbors_sum[1:, 1:] += padded_grid[:-1, :-1]  # 북서
    neighbors_sum[1:, :] += padded_grid[:-1, :]     # 북
    neighbors_sum[1:, :-1] += padded_grid[:-1, 1:]  # 북동
    neighbors_sum[:, 1:] += padded_grid[:, :-1]     # 서
    neighbors_sum[:, :-1] += padded_grid[:, 1:]     # 동
    neighbors_sum[:-1, 1:] += padded_grid[1:, :-1]  # 남서
    neighbors_sum[:-1, :] += padded_grid[1:, :]     # 남
    neighbors_sum[:-1, :-1] += padded_grid[1:, 1:]  # 남동
    
    # 패딩을 제거하여 원본 크기로 복원
    neighbors_sum = neighbors_sum[1:-1, 1:-1]
    
    # 3. 다음 세대 계산 (생명 게임 규칙 적용)
    # 살아있는 셀: 이웃이 2-3개면 살아남음, 그렇지 않으면 죽음
    # 죽은 셀: 이웃이 정확히 3개면 살아남, 그렇지 않으면 죽은 상태 유지
    next_grid = np.zeros_like(grid)
    
    # 살아있는 셀 규칙
    next_grid[(grid == 1) & ((neighbors_sum == 2) | (neighbors_sum == 3))] = 1
    
    # 죽은 셀 규칙
    next_grid[(grid == 0) & (neighbors_sum == 3)] = 1
    
    return next_grid

# 방법 2: 더 간결한 구현
def game_of_life_step_optimized(grid):
    """생명 게임의 한 스텝을 수행하는 최적화된 버전"""
    # 컨볼루션을 사용하여 이웃 합계 계산
    from scipy.signal import convolve2d
    
    # 컨볼루션 커널: 이웃을 계산하기 위한 3x3 행렬
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0  # 중앙(자기 자신)은 제외
    
    # 컨볼루션으로 이웃 개수 계산
    neighbors = convolve2d(grid, kernel, mode='same', boundary='fill', fillvalue=0)
    
    # 생명 게임 규칙 적용
    return ((grid == 1) & ((neighbors == 2) | (neighbors == 3))) | ((grid == 0) & (neighbors == 3))

# 테스트: 글라이더 패턴
def create_glider(grid, x=0, y=0):
    """글라이더 패턴을 격자에 추가"""
    glider = np.array([[0, 1, 0], 
                       [0, 0, 1], 
                       [1, 1, 1]])
    grid[x:x+3, y:y+3] = glider

# 시각화 함수
def plot_grid(grid, title=''):
    """격자를 시각화"""
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap='binary')
    plt.title(title)
    plt.grid(True)
    plt.show()

# 애니메이션 생성 함수
def animate_game_of_life(grid, steps=100, interval=100):
    """생명 게임 애니메이션 생성"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 초기 그리드 설정
    img = ax.imshow(grid, cmap='binary')
    ax.grid(True)
    
    # 애니메이션 함수
    def update(frame):
        nonlocal grid
        grid = game_of_life_step_optimized(grid)
        img.set_array(grid)
        ax.set_title(f'Step: {frame+1}')
        return [img]
    
    # 애니메이션 생성
    anim = animation.FuncAnimation(fig, update, frames=steps, interval=interval, blit=True)
    plt.close()  # 중복 표시 방지
    return anim

# 메인 실행 코드
if __name__ == "__main__":
    # 격자 크기
    N = 50
    
    # 1. 랜덤 초기 상태
    print("1. 랜덤 초기 상태로 실행")
    grid_random = np.random.choice([0, 1], size=(N, N), p=[0.8, 0.2])
    print("초기 상태:")
    print(grid_random)

    # 10세대 시뮬레이션
    print("\n10세대 시뮬레이션 실행...")
    current_grid = grid_random.copy()
    for i in range(10):
        current_grid = game_of_life_step_optimized(current_grid)
    
    print(f"10세대 후 살아있는 셀 수: {np.sum(current_grid)}")
    
    # 2. 글라이더 패턴
    print("\n2. 글라이더 패턴으로 실행")
    grid_glider = np.zeros((N, N), dtype=int)
    create_glider(grid_glider, x=10, y=10)
    print("초기 글라이더 패턴:")
    print(grid_glider[9:14, 9:14])  # 글라이더 주변만 출력
    
    # 글라이더 10세대 시뮬레이션
    current_grid = grid_glider.copy()
    for i in range(10):
        current_grid = game_of_life_step_optimized(current_grid)
    
    print("10세대 후 글라이더 패턴 (이동했을 것):")
    # 글라이더가 이동한 위치 찾기
    non_zero_indices = np.where(current_grid == 1)
    min_x, max_x = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
    min_y, max_y = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
    print(current_grid[min_x-1:max_x+2, min_y-1:max_y+2])
    
    print("\n생명 게임 성능 비교:")
    import time
    
    # 큰 격자로 성능 테스트
    grid_large = np.random.choice([0, 1], size=(1000, 1000), p=[0.9, 0.1])
    
    start = time.time()
    next_grid_1 = game_of_life_step(grid_large[:100, :100])  # 작은 부분만 테스트
    time_basic = time.time() - start
    print(f"기본 구현 시간 (100x100): {time_basic:.6f} 초")
    
    start = time.time()
    next_grid_2 = game_of_life_step_optimized(grid_large[:100, :100])
    time_optimized = time.time() - start
    print(f"최적화 구현 시간 (100x100): {time_optimized:.6f} 초")
    print(f"속도 향상: {time_basic/time_optimized:.2f}배")
    
    # 결과가 동일한지 확인
    print("두 구현의 결과가 동일함:", np.array_equal(next_grid_1, next_grid_2))

# 애니메이션 생성 코드는 노트북에서 실행할 때 사용
# 예: anim = animate_game_of_life(grid_glider, steps=50)
# HTML(anim.to_html5_video())