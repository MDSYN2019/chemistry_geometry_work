"""Exercise 18: model virtual pages and a tiny LRU page cache.

Operating systems move memory in pages. This exercise is a small simulation:
translate byte addresses to page numbers, track cache hits, and evict the
least-recently-used page when physical capacity is exceeded.
"""

from collections import OrderedDict

PAGE_SIZE_BYTES = 4096


def page_number(address: int) -> int:
    """Return the zero-based virtual page containing address."""
    # TODO: reject negative addresses with ValueError.
    # TODO: use integer floor division by PAGE_SIZE_BYTES.
    return 0


def page_offset(address: int) -> int:
    """Return the byte offset inside the page containing address."""
    # TODO: reject negative addresses with ValueError.
    # TODO: use modulo PAGE_SIZE_BYTES.
    return 0


class LruPageCache:
    def __init__(self, capacity_pages: int) -> None:
        # TODO: reject capacities less than 1.
        self.capacity_pages = capacity_pages
        self._pages: OrderedDict[int, None] = OrderedDict()

    def access(self, address: int) -> bool:
        """Access address, returning True for a cache hit and False for a miss."""
        # TODO: translate address to a page.
        # TODO: on hit, move the page to the most-recently-used end and return True.
        # TODO: on miss, add the page and evict the least-recently-used page if needed.
        return False

    def resident_pages(self) -> list[int]:
        """Return pages from least-recently-used to most-recently-used."""
        # TODO: expose the current OrderedDict keys as a list.
        return []


def simulate_cache(addresses: list[int], capacity_pages: int) -> dict[str, object]:
    """Return hit/miss counts and final resident pages."""
    # TODO: run all accesses through LruPageCache.
    return {"hits": 0, "misses": 0, "resident_pages": []}


if __name__ == "__main__":
    trace = [0, 8, 4096, 8192, 0, 4096, 12288]
    print(simulate_cache(trace, capacity_pages=3))
