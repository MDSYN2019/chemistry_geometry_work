"""Solution 18: model virtual pages and a tiny LRU page cache."""

from collections import OrderedDict

PAGE_SIZE_BYTES = 4096


def page_number(address: int) -> int:
    if address < 0:
        raise ValueError("address must be non-negative")
    return address // PAGE_SIZE_BYTES


def page_offset(address: int) -> int:
    if address < 0:
        raise ValueError("address must be non-negative")
    return address % PAGE_SIZE_BYTES


class LruPageCache:
    def __init__(self, capacity_pages: int) -> None:
        if capacity_pages < 1:
            raise ValueError("capacity_pages must be at least 1")
        self.capacity_pages = capacity_pages
        self._pages: OrderedDict[int, None] = OrderedDict()

    def access(self, address: int) -> bool:
        page = page_number(address)
        if page in self._pages:
            self._pages.move_to_end(page)
            return True
        self._pages[page] = None
        if len(self._pages) > self.capacity_pages:
            self._pages.popitem(last=False)
        return False

    def resident_pages(self) -> list[int]:
        return list(self._pages.keys())


def simulate_cache(addresses: list[int], capacity_pages: int) -> dict[str, object]:
    cache = LruPageCache(capacity_pages)
    hits = 0
    misses = 0
    for address in addresses:
        if cache.access(address):
            hits += 1
        else:
            misses += 1
    return {"hits": hits, "misses": misses, "resident_pages": cache.resident_pages()}


if __name__ == "__main__":
    trace = [0, 8, 4096, 8192, 0, 4096, 12288]
    print(simulate_cache(trace, capacity_pages=3))
