# Exercise List (with solution mapping)

## Core Python fundamentals

| # | Topic | Exercise File | Solution File |
|---|---|---|---|
| 00 | f-strings and joins | `exercises/00_string_builder/string_builder.py` | `solutions/00_string_builder.py` |
| 01 | conditionals and loops | `exercises/01_control_flow/control_flow.py` | `solutions/01_control_flow.py` |
| 02 | list/dict transformations | `exercises/02_collections/collections_ops.py` | `solutions/02_collections_ops.py` |
| 03 | function defaults and kwargs | `exercises/03_functions/keyword_report.py` | `solutions/03_keyword_report.py` |

## Async foundations

| # | Topic | Exercise File | Solution File |
|---|---|---|---|
| 04 | asyncio basics (`async`/`await`) | `exercises/04_async_sleep/async_sleep.py` | `solutions/04_async_sleep.py` |
| 05 | async concurrency with `gather` | `exercises/05_async_gather/async_gather.py` | `solutions/05_async_gather.py` |

## Data modeling and validation

| # | Topic | Exercise File | Solution File |
|---|---|---|---|
| 06 | dataclasses intro | `exercises/06_dataclasses_intro/student_record.py` | `solutions/06_student_record.py` |
| 07 | staticmethod factories | `exercises/07_staticmethod_factory/temperature.py` | `solutions/07_temperature.py` |
| 08 | pydantic model validation | `exercises/08_pydantic_models/order_model.py` | `solutions/08_order_model.py` |
| 09 | dataclass + staticmethod + pydantic | `exercises/09_dataclass_staticmethod_pydantic/inventory_bridge.py` | `solutions/09_inventory_bridge.py` |

## Object orientation methods (basic to advanced)

| # | Topic | Exercise File | Solution File |
|---|---|---|---|
| 10 | instance methods and class methods | `exercises/10_instance_class_methods/bank_account.py` | `solutions/10_instance_class_methods.py` |
| 11 | dunder methods (`__repr__`, `__add__`, `__eq__`) | `exercises/11_dunder_methods/vector2d.py` | `solutions/11_dunder_methods.py` |
| 12 | abstract methods and polymorphism (ABC) | `exercises/12_abstract_methods/shape_pricing.py` | `solutions/12_abstract_methods.py` |
| 13 | mixins and `super()` method reuse | `exercises/13_mixin_super_calls/notifications.py` | `solutions/13_mixin_super_calls.py` |

## Practical data engineering patterns

| # | Topic | Exercise File | Solution File |
|---|---|---|---|
| 10 | data modeling + contracts | `exercises/10_data_modeling_contracts/data_contracts.py` | `solutions/10_data_contracts.py` |
| 11 | transformations + partitioning | `exercises/11_transformation_partitioning/partition_transform.py` | `solutions/11_partition_transform.py` |
| 12 | orchestration + SLA reporting | `exercises/12_orchestration_slas/orchestrate_pipeline.py` | `solutions/12_orchestrate_pipeline.py` |
| 13 | cloud-native architecture choices | `exercises/13_cloud_native_architecture/cloud_architecture.py` | `solutions/13_cloud_architecture.py` |
| 14 | governance, lineage, and access checks | `exercises/14_governance_lineage_access/governance_checks.py` | `solutions/14_governance_checks.py` |
| 15 | cost optimization + performance tuning | `exercises/15_cost_perf_tuning/cost_perf_tuning.py` | `solutions/15_cost_perf_tuning.py` |

## Computer science systems fundamentals

| # | Topic | Exercise File | Solution File |
|---|---|---|---|
| 16 | `__slots__` and per-instance memory overhead | `exercises/16_slots_memory/profile_records.py` | `solutions/16_profile_records.py` |
| 17 | choosing `deque` vs `list` for queues and rotation | `exercises/17_deque_work_queue/work_queue.py` | `solutions/17_work_queue.py` |
| 18 | virtual pages and LRU memory pressure simulation | `exercises/18_memory_pages/page_cache.py` | `solutions/18_page_cache.py` |
| 19 | kernel-style round-robin scheduling and time slices | `exercises/19_scheduler_timeslices/scheduler.py` | `solutions/19_scheduler.py` |

## Object-Oriented Python sushi simulator

These exercises mirror a four-day OOP course arc: start with plain data that wants to become objects, then add Python's data model, design judgment around inheritance vs. composition, and finally properties/dataclasses/SOLID-style swappable behavior.

| # | Topic | Exercise File | Solution File |
|---|---|---|---|
| 20 | classes, instances, `__repr__`, and mutable attribute traps | `exercises/20_oop_sushi_day1/sushi_plate.py` | `solutions/20_sushi_plate.py` |
| 21 | data model methods, sequence behavior, comparison, and alternate constructors | `exercises/21_oop_sushi_data_model/conveyor_belt.py` | `solutions/21_conveyor_belt.py` |
| 22 | `@classmethod`, inheritance, composition, ABCs, and polymorphism | `exercises/22_oop_sushi_creation_design/restaurant_design.py` | `solutions/22_restaurant_design.py` |
| 23 | `@property`, dataclasses, `default_factory`, protocols, and swappable pricing | `exercises/23_oop_sushi_properties_solid/simulator.py` | `solutions/23_simulator.py` |

