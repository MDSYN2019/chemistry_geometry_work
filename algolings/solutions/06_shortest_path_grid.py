from collections import deque


def shortest_path_grid(grid: list[list[int]], start: tuple[int, int], goal: tuple[int, int]) -> int:
    rows, cols = len(grid), len(grid[0])
    sr, sc = start
    gr, gc = goal
    if grid[sr][sc] == 1 or grid[gr][gc] == 1:
        return -1

    q = deque([(sr, sc, 0)])
    seen = {(sr, sc)}
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))

    while q:
        r, c, dist = q.popleft()
        if (r, c) == (gr, gc):
            return dist
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0 and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc, dist + 1))
    return -1
