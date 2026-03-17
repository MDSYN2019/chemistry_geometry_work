# Exercise List (with solution mapping)

| # | Topic | Exercise File | Solution File |
|---|---|---|---|
| 00 | Use reduce to sum a list of integers. | `exercises/00_reduce_basics/sum_with_reduce.py` | `solutions/00_sum_with_reduce.py` |
| 01 | Use reduce to compute product with initializer 1. | `exercises/01_reduce_product/product_with_reduce.py` | `solutions/01_product_with_reduce.py` |
| 02 | Use reduce to compute max value. | `exercises/02_reduce_max/max_with_reduce.py` | `solutions/02_max_with_reduce.py` |
| 03 | Use partial to build square and cube from a power function. | `exercises/03_partial_intro/make_power_functions.py` | `solutions/03_make_power_functions.py` |
| 04 | Use partial keyword args for a log formatter. | `exercises/04_partial_keywords/configure_logger.py` | `solutions/04_configure_logger.py` |
| 05 | Compose partial callables for text normalization. | `exercises/05_partial_pipeline/run_pipeline.py` | `solutions/05_run_pipeline.py` |
| 06 | Use wraps in a decorator preserving metadata. | `exercises/06_wraps_basics/trace.py` | `solutions/06_trace.py` |
| 07 | Use update_wrapper on a callable class wrapper. | `exercises/07_update_wrapper/build_wrapper.py` | `solutions/07_build_wrapper.py` |
| 08 | Sort tuples by age then name with cmp_to_key. | `exercises/08_cmp_to_key_sort/sort_people.py` | `solutions/08_sort_people.py` |
| 09 | Sort semantic-ish versions with cmp_to_key. | `exercises/09_cmp_to_key_custom/sort_versions.py` | `solutions/09_sort_versions.py` |
| 10 | Memoize recursive fibonacci with lru_cache. | `exercises/10_lru_cache_fib/fib.py` | `solutions/10_fib.py` |
| 11 | Cache function with typed=True behavior demo. | `exercises/11_lru_cache_typed/add_one.py` | `solutions/11_add_one.py` |
| 12 | Expose cache_info and cache_clear usage. | `exercises/12_lru_cache_control/expensive_lookup.py` | `solutions/12_expensive_lookup.py` |
| 13 | Use @cache for pure function scoring strings. | `exercises/13_cache_decorator/word_score.py` | `solutions/13_word_score.py` |
| 14 | Create cached_property area on a class. | `exercises/14_cached_property_intro/Circle.py` | `solutions/14_Circle.py` |
| 15 | Invalidate cached_property after mutation. | `exercises/15_cached_property_invalidation/UserProfile.py` | `solutions/15_UserProfile.py` |
| 16 | Dispatch by input type using singledispatch. | `exercises/16_singledispatch_basics/to_jsonable.py` | `solutions/16_to_jsonable.py` |
| 17 | Dispatch list/tuple/set handling. | `exercises/17_singledispatch_collections/flatten_once.py` | `solutions/17_flatten_once.py` |
| 18 | Use singledispatchmethod in class. | `exercises/18_singledispatchmethod_intro/Formatter.py` | `solutions/18_Formatter.py` |
| 19 | Use partialmethod for preset greeting styles. | `exercises/19_partialmethod_intro/Greeter.py` | `solutions/19_Greeter.py` |
| 20 | Use total_ordering with eq and lt. | `exercises/20_total_ordering_intro/Temperature.py` | `solutions/20_Temperature.py` |
| 21 | Use total_ordering for sortable task priorities. | `exercises/21_total_ordering_sort/TaskPriority.py` | `solutions/21_TaskPriority.py` |
| 22 | Reduce dataclass items into total. | `exercises/22_reduce_with_dataclasses/Invoice.py` | `solutions/22_Invoice.py` |
| 23 | Cache normalized tokens per input text. | `exercises/23_lru_cache_methods/Tokenizer.py` | `solutions/23_Tokenizer.py` |
| 24 | Demonstrate eviction with maxsize=2. | `exercises/24_lru_cache_eviction/rolling_result.py` | `solutions/24_rolling_result.py` |
| 25 | Decorator preserving annotations/docstring. | `exercises/25_wraps_annotations/validate_positive.py` | `solutions/25_validate_positive.py` |
| 26 | Register multiple concrete implementations. | `exercises/26_singledispatch_register/render.py` | `solutions/26_render.py` |
| 27 | Factory returning partial multiplier callables. | `exercises/27_partial_class_factory/build_multiplier.py` | `solutions/27_build_multiplier.py` |
| 28 | Case-insensitive, article-agnostic sort comparator. | `exercises/28_cmp_to_key_locale/sort_titles.py` | `solutions/28_sort_titles.py` |
| 29 | cached_property on frozen-like analysis class. | `exercises/29_cached_property_dataclass/Experiment.py` | `solutions/29_Experiment.py` |
| 30 | Reduce iterable into frequency dict. | `exercises/30_reduce_grouping/count_letters.py` | `solutions/30_count_letters.py` |
| 31 | Combine partial, singledispatch, and lru_cache. | `exercises/31_capstone_mini_toolkit/run_demo.py` | `solutions/31_run_demo.py` |
