import heapq


def dijkstra(graph: list[list[tuple[int, int]]], source: int, target: int) -> int:
    dist = [float("inf")] * len(graph)
    dist[source] = 0
    heap: list[tuple[int, int]] = [(0, source)]

    while heap:
        d, node = heapq.heappop(heap)
        if d > dist[node]:
            continue
        if node == target:
            return d
        for nxt, w in graph[node]:
            nd = d + w
            if nd < dist[nxt]:
                dist[nxt] = nd
                heapq.heappush(heap, (nd, nxt))

    return -1
