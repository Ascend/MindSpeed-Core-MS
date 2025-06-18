## stack\_utils 调试堆栈使用说明

本说明介绍如何在单机或分布式场景中，对任意位置堆栈的打印，以及对指定函数（名称或正则模式）调用时的堆栈跟踪。

---

### 目录

1. [功能概述](#功能概述)
2. [使用方式](#使用方式)
3. [示例](#示例)
4. [API 参考](#API-参考)
5. [原理与性能](#原理与性能)
6. [常见问题](#常见问题)

---

### 功能概述

* **任意位置打印堆栈**
  调用 `show_stack()` 即可在脚本任何一处打印当前完整调用链。
* **上下文管理器/装饰器模式**
  `with TraceContext(...):` 进入/退出或指定函数调用时自动打印堆栈，也可通过 `@TraceContext(...)` 装饰函数。
* **严格字符串匹配 & 正则模式 & 混合匹配**
  支持传入函数名列表（严格字符串匹配）和正则列表（正则匹配），或两者组合。
* **默认模式**
  未传任何参数时，`with TraceContext():` 会在进入和退出被装饰/被包裹的函数时，各打印一次堆栈，不影响原有函数执行性能。

---

### 使用方式

1. 将下面的 `stack_utils.py` 放入项目目录，如 `utils/stack_utils.py`。
2. 在脚本中引入：

   ```python
   from tools import show_stack, TraceContext
   ```

---

### 示例

#### 1. 任意位置打印堆栈

```python
from tools import show_stack

def foo():
    bar()

def bar():
    # 在这里打印当前堆栈
    show_stack()

foo()
```

#### 2. 默认模式：进入 & 退出上下文

```python
from tools import TraceContext

print("进入 default 上下文")
with TraceContext():
    # __enter__ 时打印一次堆栈
    pass
# __exit__ 时再打印一次堆栈
```

#### 3. 严格字符串匹配

```python
from tools import TraceContext

def foo(): pass
def init(): pass

with TraceContext('foo', 'init'):
    foo()   # 打印
    init()  # 打印
```

#### 4. 正则模式

```python
from tools import TraceContext

def foo1(): pass
def foo2(): pass

# 只匹配名称完全符合正则 foo[0-9]+ 的函数
with TraceContext(regex=[r'foo[0-9]+']):
    foo1()  # 打印
    foo2()  # 打印
```

#### 5. 混合模式

```python
from tools import TraceContext

def foo1(): pass
def init(): pass

with TraceContext('init', regex=[r'foo[0-9]+']):
    init()  # 严格字符串匹配，打印
    foo1()  # 正则匹配，打印
```

#### 6. 装饰器模式

```python
from tools import TraceContext

@TraceContext('myprefix_.*', regex=[r'.*_handler'])
def my_handler():
    pass

# 每次调用 my_handler 时都会打印堆栈
my_handler()
```

---

### API 参考

#### `show_stack()`

* **功能**：打印当前调用链（包含自身调用）。
* **用法**：

  ```python
  show_stack()
  ```

#### `TraceContext(*literal_names, regex: List[str]=None)`

1. **构造参数**
  * `*literal_names`：要跟踪的函数名列表（精确匹配）。
  * `regex`：要跟踪的正则表达式列表（名称需完全匹配正则）。

2. **上下文管理器用法**

  ```python
  with TraceContext(...):
      ...
  ```

3. **装饰器用法**

  ```python
  @TraceContext(...)
  def func(...):
      ...
  ```

---

### 原理与性能

1. **原理**
  * 跟踪模式：在 `__enter__` 时通过 `sys.settrace()` 安装自定义 tracer，对每个 `call` 事件判断是否匹配严格字符串名称或正则， 匹配时打印堆栈；在 `__exit__` 卸载 tracer。

2. **性能注意**
  * **字符串精准匹配**：基于 `set` 哈希判断，O(1) 开销，可忽略。
  * **正则匹配**：仅对少量模式做 `fullmatch`，对于每个函数都会进行匹配，可能导致明显延迟。
  * **默认模式**：无全局追踪，仅在两次上下文边界打印一次，性能开销极低。

---

### 常见问题

1. **“为什么没打印？”**
  * 确认 `with TraceContext(...)` 块内是否真正调用了匹配函数；
  * 若使用装饰器，请确保装饰过的函数被调用。

2. **“多次打印堆栈，怎样只看最新一次？”**
  * 可在调用 `show_stack()` 或进入/退出上下文前后插入分隔日志；
  * 也可自定义 `_print()` 方法，过滤重复输出。
