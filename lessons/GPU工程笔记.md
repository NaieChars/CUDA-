---
title: GPU工程实践笔记-C++/CUDA/Powershell
published: 2026-10-03
pinned: true
description: 记录可复用的工程设计、资源管理、错误处理、命令行参数解析与自动化实践。
tags: [CUDA, C++, Powershell]
category: 技术
draft: false
---

[查看源码](https://github.com/NaieChars/CUDA-/blob/main/labs/sgemm/src/main.cu)

# C++ 部分

> [!NOTE]
> 在阅读本文前，你需要掌握 C++ 的基本语法，并了解一些 C++ 的常用工程技巧，包括但不限于：接口与实现分离，RAII，工厂模式，配置模式……
> 如果你忘记了一些常用语法，部分语法在文中也会略有提及。

本文以 `labs/sgemm/src/main.cu` 为例，整理了 CUDA 实验框架中可复用的 C++ 工程思路。主要讲述的是**各部分的分工设计**，以及**关键语法**如何支撑这些设计。以后的跑分实验均可沿用此模版，有很好的学习意义。

> 学习进度（2026-10-20，第六讲）：先转入 PowerShell，学习路径、命令调用、参数、数组、引号、变量与续行；之后继续脚本内部语法和 CUDA 头文件中的 RAII / 检错宏 / 计时设计。C++ 语法仅聚焦尚不熟悉的关键写法。本节通过无 GPU 的 --plan 和独立参数绑定示例核对基础规则，未执行跑分脚本。

# 1. 基本架构

## 1.1 main 控制流程

```cpp
try { return run(parse_options(argc, argv)); }
```

这里先执行 `parse_options`，成功返回配置后才调用 `run`。

**主控制流程**如下：

```text
命令行文本
    ↓ parse_options：转换类型，检查参数约束
Options 配置
    ↓ run：选择路径并组织实验
    ├─ help / list / plan：输出信息后返回
    └─ GPU 实验：准备数据 → 启动核函数 → 校验 → 测量或采集
```

`--help` 等分支位于 `run` 内，也会先经过 `parse_options`，即帮助参数不会跳过所有参数检查

| 模块 | 当前源码位置 | 主要职责 |
| --- | --- | --- |
| 配置及参数检查 | `main.cu` 的 `Options`、`parse_*`、`matrix_elements` | 将文本转为配置，检查数值与组合约束 |
| 正确性检查 | `main.cu` 的 `validate` | 比较 GPU 输出与 CPU double 参考值 |
| 实验编排 | `main.cu` 的 `run` | 查询设备、准备数据、选择检查/计时/采集路径 |
| 程序入口 | `main.cu` 的 `main` | 连接解析与执行，处理传播到这里的标准异常 |
| 算子接口与实现 | `kernels.cuh` / `kernels.cu` | 描述变体、计算启动配置、分派并实现核函数 |
| 公共基础设施 | `common/include/cuda_*.cuh` | 管理显存、CUDA 检错和事件计时 |

## 1.2 匿名命名空间

> **作用**：把实现细节留在本翻译单元

```cpp
namespace
{
constexpr int kInvalidArguments = 2;
constexpr int kNotImplemented = 3;
constexpr int kValidationFailed = 4;
// Options、解析函数、validate、run 等也在其中。
}
```

没有名字的 `namespace` 称为匿名命名空间。这里用它收纳**本翻译单元**（一个源文件经过预处理、展开头文件后交给编译器的内容）的内部实现，避免把通用名字（如 `run`、`Options`）作为跨文件接口暴露出去。

`constexpr int` 在这里声明可用于常量表达式的整数常量；`k` 前缀只是常量命名习惯。**用 `kValidationFailed` 表达返回原因**，比散落的 `return 4` 更容易读懂。

## 1.3 Options

> **作用**：把一次实验的配置组织成一个值，类似于<u>配置模式</u>。

```cpp
struct Options
{
    int mode = 0;
    size_t m = 512, n = 512, k = 512;
    int warmup = 10;
    int iterations = 100;
    double atol = 1e-4, rtol = 1e-4;
    bool help = false, list = false, plan_only = false;
    bool check_only = false, profile = false;
};
```

与 `class` 类似，执行 `Options options;` 时，这里的各成员会使用相应默认值，不需要额外写一个只负责赋值的构造函数。

实验配置组合成 `Options` 后，解析器负责构造配置，执行函数通过 `run(const Options& options)` 读取配置。相较于一长串独立参数，调用更清楚，新增配置项时也更容易维护。

注意：`Options` 本身没有强制所有约束。例如仍可手动写入 `options.iterations = 0`。当前程序依靠 `parse_options` 检查其负责的输入约束，`run` 再检查设备相关限制；“配置结构”不等于“永远不可能产生非法状态的类型”。

# 2. 针对命令行参数错误的处理

## 2.1 检测命令行字符串合规性

### 2.1.1 严格解析无符号十进制整数

#### 核心实现

```cpp
size_t parse_unsigned(const std::string& text, const char* name)
{
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument(std::string(name) + " must be an unsigned integer");

    const auto value = std::stoull(text);
    if (value > std::numeric_limits<size_t>::max())
        throw std::invalid_argument(std::string(name) + " is too large");

    return static_cast<size_t>(value);
}
```

这个函数承诺：成功时返回一个能由 `size_t` 表示的非负整数；失败时抛出异常。

内部顺序是：**检查文本格式 → 转换为中间整数 → 检查目标类型范围 → 转为目标类型**。转换成功，只能说明标准库能够提取数值，不能自动证明它符合本程序的输入规则。

#### 参数：输入内容与错误上下文分开

```cpp
parse_unsigned("512", "M");
parse_unsigned("abc", "warmup"); // 报错会带上 warmup
```

- `const std::string& text`：**以只读引用接收字符串**。已有 `std::string` 传入时不必复制字符串；若传入 `argv[i]` 这样的 C 字符串，**仍需先隐式构造临时 `std::string`**，临时对象可绑定到这个引用。
- `const char* name`：借用一个以空字符结尾的字符序列，用于说明哪个参数出错。这里不取得其所有权；`const` 修饰所指字符，并不表示指针变量本身不可改指向。
- `size_t`：我在 Windows 上表示的是 64 位无符号整数，具体位宽跟系统有关。

#### empty、find_first_not_of 与 npos

`text.empty()` 检查是否为空。

`text.find_first_not_of("0123456789")` 从头寻找第一个不属于字符集合 `0`～`9` 的字符，**返回第一个非法字符位置**。

| 文本 | 查找结果 | 含义 |
| --- | --- | --- |
| `"512"` | **`std::string::npos`** | 没有找到非数字字符 |
| `"-1"` | `0` | 第一个字符就不属于集合 |
| `""` | **`std::string::npos`** | 没有字符，因此也找不到违规字符 |

`std::string::npos` 是字符串类型提供的静态成员常量，此处表示**没有找到**。因此 `!= npos` 表示确实找到了违规字符[标准草案：字符串查找](https://eel.is/c++draft/string.find)、[查找条件定义](https://eel.is/c++draft/string.view.find)

空字符串同样会返回 `npos`，所以需要单独用 `empty()` 拒绝它。

#### `stoull`

```cpp
const auto value = std::stoull(text);
// 此处相当于：
// const unsigned long long value = std::stoull(text);
```

`std::stoull` 是 `<string>` 提供的自由函数，返回 `unsigned long long`；`auto` 让编译器根据初始化表达式推导类型，并没有让变量变成动态类型。

`std::stoull` 可以只转换字符串开头的有效部分，例如直接调用 `std::stoull("12x")` 可返回 12。因此，本程序先要求全文都是数字。这里的 `unsigned` 也不能替代文本规则检查。

> [!NOTE]
若纯数字文本表示的值连 `unsigned long long` 都放不下，`std::stoull` 会**抛出** **`std::out_of_range`**，不会执行后面的 `size_t` 范围检查。**即不是所有错误都是本函数手动抛出的 `std::invalid_argument`**。[标准草案：数值转换](https://eel.is/c++draft/string.conversions)
>

#### numeric_limits 与 static_cast

```cpp
std::numeric_limits<size_t>::max()
```

`numeric_limits<size_t>` 指定查询 `size_t` 特征的类模板特化，`::max()` 调用它的静态成员函数。它返回该类型的最大可表示值，不需要先创建一个对象。[标准草案：数值类型特征](https://eel.is/c++draft/numeric.limits.members)

`static_cast<size_t>(value)` 显式表达转换意图，但它**本身不负责抛出越界异常**。

#### throw 与字符串拼接
> **`throw` 的核心思想：底层描述错误，上层决定处理方式**

```cpp
throw std::invalid_argument(std::string(name) + " is too large");
```

`std::string(name)` **先构造字符串对象**（很容易忘记，直接用 C 字符串指针相加），让后面执行字符串拼接

`throw` 抛出异常后，这条执行路径立即离开当前函数，沿调用链寻找匹配的 `catch`。当前文件的入口负责接住传播到这里的标准异常：

```cpp
// main()...
catch (const std::exception& error)
{
    std::fprintf(stderr, "Error: %s\nUse --help for usage.\n", error.what());
    return kInvalidArguments;
}
```

`std::invalid_argument` 和 `std::out_of_range` 都派生自 `std::exception`，可被这里的基类引用接住。`error.what()` 读取异常的描述文字。

这样，**解析函数负责发现并描述错误，入口负责面向用户输出和选择退出码**。这种职责划分**便于复用解析函数，也便于单独检查解析行为**。

不过，当前的 `CUDA_CHECK` 宏走的是直接打印并 `exit` 的路径，不经过此 `catch`。异常传播时的栈展开与直接 `exit` 的资源清理行为也不同，不能把两者视为同一种错误处理机制。

  > [!WARNING]
> 该函数只是检测格式是否有效，格式有效不代表业务有效。由于在我的 CUDA 性能实验中，矩阵的维度必须大于 0，而这项条件束缚是后续 `matrix_elements` 检查的业务约束。所以这里体现了一个关键的设计思路：**将格式、类型范围和业务约束分层检查，并让错误携带调用上下文**。

### 2.1.2 复用已有解析逻辑，收紧到 int 范围
#### 核心实现

```cpp
int parse_int(const std::string& text, const char* name)
{
    const size_t value = parse_unsigned(text, name);
    if (value > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument(std::string(name) + " is too large");
    return static_cast<int>(value);
}
```

这个函数在已有解析能力上增加一层限制：先复用 `parse_unsigned` 的文本与范围检查，再确认数值不超过 `int` 的最大值，最后转换。

`static_cast<size_t>(std::numeric_limits<int>::max())` 将比较上界显式转换到与 `value` 相同的类型。在本项目的平台上，`size_t` 可以表示这个正整数上界。比较通过后，再将 `value` 转成 `int`。

> [!NOTE]
> 名称里的 `int` 表示返回类型。由于复用了 `parse_unsigned`，它实际只接受非负整数，并不是一个接受正负号的通用 int 解析器。

为什么这些配置用 int？`mode` 是少量模式编号，`warmup`、`iterations` 则用于 int 循环计数，返回类型与使用处保持一致即可。矩阵大小与下标继续使用适合表达大小的 `size_t`。

**设计思维**：每层复用下层已经建立的保证，只增加自己负责的约束。例如 `warmup = 0` 可以有效，而 `iterations = 0` 会由后续 `parse_options` 拒绝；这类业务差异不必塞进通用解析函数。

### 2.1.3 浮点容差：完整转换、有限、非负

```cpp
double parse_tolerance(const std::string& text)
{
    size_t used = 0;
    const double value = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(value) || value < 0)
        throw std::invalid_argument("tolerances must be finite and nonnegative");
    return value;
}
```

浮点文本可以包含小数点、指数等形式，例如 `0.001`、`1e-4`。这里让标准库负责数值语法，再根据本程序的需求检查转换结果。

#### 一个返回值，加一个输出参数

`std::stod` 返回转换得到的 double；第二个参数的类型是 `size_t*`，用于写回第一个未转换字符的下标。

```cpp
size_t used = 0;
double value = std::stod("0.01abc", &used);
// value 为 0.01，used 为 4。
```

`&used` 取得变量地址，让函数通过指针修改调用者的变量。这是**输出参数**：一个普通的局部变量即可接收结果，不涉及动态分配。

`used == text.size()` 表示没有剩余字符；`used != text.size()` 则拒绝只转换成功一部分的文本。注意 `used` 是停止位置，不是有效数字的个数。[标准草案：字符串数值转换](https://eel.is/c++draft/string.conversions)

如果文本根本无法转换，`stod` 会直接抛出 `invalid_argument`；如果转换报告超出范围，则抛出 `out_of_range`，不会继续执行后面的 if。

#### 三种检查各自保证什么

| 条件 | 拒绝的情况 | 例子 |
| --- | --- | --- |
| `used != text.size()` | 有未转换的尾部字符 | `"0.01abc"` |
| `!std::isfinite(value)` | NaN 或正负无穷 | `"nan"`、`"inf"` |
| `value < 0` | 负容差 | `"-0.01"` |

`std::isfinite` 检查数值是否有限；零属于有限值。仅判断 `< 0` 无法拒绝 NaN 和正无穷，因此需要独立检查。[Microsoft：cmath 分类函数](https://learn.microsoft.com/en-us/cpp/standard-library/cmath?view=msvc-170)

这里也有输入规则上的取舍：`stod` 可以跳过前导空白，因此 `" 0.01"` 可以通过当前函数，而 `"0.01 "` 留下尾部空白，会被完整性检查拒绝。前导加号、科学计数法也可接受；它没有沿用上一节“全文只能是数字”的规则。[Microsoft：strtod 的转换规则](https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/strtod-strtod-l-wcstod-wcstod-l?view=msvc-170)

**设计思维**：标准库负责解释数值文本，程序负责规定结果应满足的条件。容差需要是有限非负数，才能用于后续数值比较。

## 2.2 矩阵大小：在乘法发生前防止溢出

```cpp
size_t matrix_elements(size_t rows, size_t columns, size_t element_bytes)
{
    const size_t limit = std::numeric_limits<size_t>::max();
    if (rows == 0 || columns == 0 || rows > limit / columns)
        throw std::invalid_argument("matrix dimensions must be positive and fit size_t");
    const size_t count = rows * columns;
    if (count > limit / element_bytes)
        throw std::invalid_argument("matrix byte size does not fit size_t");
    return count;
}
```

### 2.2.1 为什么不能先乘，再检查

```cpp
// 错误思路：
if (rows * columns > limit) { /* 报错 */ }
```

`rows`、`columns` 都是 `size_t`，乘积也按该无符号类型计算。超出范围时结果按模回绕，比较看到的已经不是数学上的真实乘积。

因此，无符号乘积不会通过 `> size_t最大值` 来暴露问题。无符号回绕是语言规定的行为，但会破坏“需要分配多少元素”的业务含义。[标准草案：无符号整数运算](https://eel.is/c++draft/basic.fundamental)

### 2.2.2 把乘法约束改写成除法约束

对正整数 `columns`，希望满足：

```text
rows × columns ≤ limit
等价于
rows ≤ floor(limit / columns)
```

因此，代码先检查 `rows > limit / columns`，通过后才计算乘积。整数除法自然向下取整，正好给出允许的最大行数。

假设一种类型最大值为 255：`rows = 20`、`columns = 13` 时，真实乘积为 260；先检查 `20 > 255 / 13`，也就是 `20 > 19`，就可以提前拒绝。这里是便于推导的假想范围，不是本机 size_t 的宽度。

### 2.2.3 短路求值保护除法

```cpp
rows == 0 || columns == 0 || rows > limit / columns
```

这里条件的排列具有安全意义：当 `columns == 0` 时，内建 `||` 不再计算后面的除法，从而避免除零。检查前置条件必须发生在使用这个条件的运算之前。[标准草案：逻辑或](https://eel.is/c++draft/expr.log.or)

### 2.2.4 元素数与字节数是两层约束

第一道检查保证 `rows * columns` 能表示；第二道检查保证 `count * element_bytes` 也能表示。

元素数装得下，不代表再乘 `sizeof(float)` 或 `sizeof(double)` 后的字节数仍装得下。即使函数最后返回的是元素个数，调用者分配和拷贝内存时也需要安全的字节数。

```cpp
matrix_elements(options.m, options.n, sizeof(double));
```

解析阶段对 C 的形状使用 `sizeof(double)`，对应 CPU double 参考结果的大小；GPU 的 float 输出需要的字节数更少。`sizeof(T)` 返回类型 T 占用的字节数，类型形式不需要创建对象。

> [!NOTE]
> 当前调用处都传入 `sizeof(float)` 或 `sizeof(double)`，保证 `element_bytes > 0`。函数自身没有检查第三个参数是否为零，因此不能假设它能安全接受任意 size_t 参数组合。

检查通过仅表示**大小的算术表达没有超出 size_t 范围**，并不保证主机内存、显存或容器容量足够，实际分配仍可能失败。

本节待自测：若某整数类型最大值为 255，矩阵形状为 10×10，每元素 4 字节，两道检查各自会怎样？解释时区分元素数与字节数。

## 2.3 parse_options：从零散文本组装实验配置

前面的解析函数检查一个数值，这个函数则负责识别参数角色，并检查一次实验的整体配置。

### 2.3.1 先看两阶段结构

```text
Options 默认值 + argv 文本序列
    ↓ 第一阶段：扫描参数
    ├─ 无值开关：设置布尔成员
    ├─ 带值选项：消费下一个参数，转换后设置成员
    └─ 位置参数：按原顺序暂存到 positional
    ↓ 第二阶段：解释位置参数并检查整体约束
    mode / M / N / K → 检查模式、矩阵大小、迭代次数、选项互斥
    ↓
按值返回 Options
```

这种分工允许开关穿插在位置参数之间；位置参数彼此的相对顺序仍必须是 `mode M N K`。本节先解释 C++ 的解析实现，终端引号、调用与脚本语法留到后续统一整理。

### 2.3.2 argv 文本与 vector 的职责

```cpp
Options options;
std::vector<std::string> positional;
for (int i = 1; i < argc; ++i)
{
    const std::string arg = argv[i];
    // 按参数角色处理 arg。
}
```

常见调用中，`argv[0]` 是程序调用名称，`argv[1]` 起才是用户提供的参数；`argc` 包含程序名称这一项。`char** argv` 可以理解为指向一组 C 字符串指针，`argv[i]` 取出其中一个以空字符结尾的字符串。[标准草案：main 参数](https://eel.is/c++draft/basic.start.main)

`const std::string arg = argv[i]` 构造一个拥有自身内容的字符串，方便使用字符串成员函数和按内容比较的 `==`。它还保留当前参数的内容，后续即使修改 i，arg 也仍表示原来的选项名。

`std::vector<std::string>` 是字符串的动态数组。遇到位置参数时，`positional.push_back(arg)` 将字符串复制到末尾，保留出现顺序；循环内 arg 离开作用域不会让已保存的字符串悬空。`size()` 取数量，`empty()` 判断是否为空，`operator[]` 按下标访问；下标访问本身不做越界检查，因此后文先检查数量再访问。[标准草案：vector 插入](https://eel.is/c++draft/vector.modifiers)

### 2.3.3 带值选项：一次分支消费两个参数

```cpp
else if (arg == "--warmup" || arg == "--iterations" ||
         arg == "--atol" || arg == "--rtol")
{
    if (++i >= argc) throw std::invalid_argument(arg + " needs a value");
    if (arg == "--warmup") options.warmup = parse_int(argv[i], "warmup");
    else if (arg == "--iterations") options.iterations = parse_int(argv[i], "iterations");
    else if (arg == "--atol") options.atol = parse_tolerance(argv[i]);
    else options.rtol = parse_tolerance(argv[i]);
}
```

以这组收到的文本为例：

```text
下标：  0       1    2       3        4    5    6         7
参数：sgemm     1   128   --warmup    2   64   32   --check-only
```

| i 的位置 | 本轮动作 |
| --- | --- |
| 1、2 | 将 `"1"`、`"128"` 存入 positional |
| 3 | 保存 arg 为 `"--warmup"`；`++i` 移到 4，解析 `"2"` |
| 本轮结束 | for 自身的 `++i` 将 i 推进到 5，不再把 `"2"` 当成位置参数 |
| 5、6 | 将 `"64"`、`"32"` 存入 positional |
| 7 | 将 check_only 设为 true |

最终 `positional` 是 `{"1", "128", "64", "32"}`，对应 mode=1、M=128、N=64、K=32；warmup=2，check_only=true。

`if (++i >= argc)` 先推进索引，再判断有没有值可读。若选项位于末尾，抛异常发生在访问 `argv[i]` 之前。

最后的 `else options.rtol = ...` 依赖外层已将 arg 限定在四个名称中：排除前三个后，只可能剩下 `--rtol`。如果以后新增带值选项，必须一起维护这个分派逻辑。

**设计思维**：把 i 当作读取游标，分支负责消费本参数额外需要的值，外层循环负责推进到下一项。边界检查要发生在读取之前。

### 2.3.4 rfind 的特殊用法：只检查前缀

```cpp
else if (arg.rfind("--", 0) == 0)
    throw std::invalid_argument("unknown option: " + arg);
else positional.push_back(arg);
```

`rfind` 的第二个参数限制匹配起始位置的上界。传入 0 后，只有从下标 0 开始的匹配才可能成功，因此这里等价于判断字符串是否以 `--` 开头。[标准草案：rfind 查找条件](https://eel.is/c++draft/string.view.find)

| arg | `rfind("--", 0)` | 这个条件的结果 |
| --- | --- | --- |
| `"--typo"` | 0 | 识别为未知长选项 |
| `"abc--def"` | npos | 不满足前缀条件 |
| `"128"` | npos | 不满足前缀条件 |

已知选项的分支必须位于这道检查之前，否则合法的 `--warmup` 等也会被当成未知选项。

**设计思维**：明确拒绝拼错的长选项，让用户尽早得到错误信息。位置参数留给后续阶段解释。

### 2.3.5 位置参数与组合约束

```cpp
if (positional.size() > 4)
    throw std::invalid_argument("expected: mode M N K");
if (!positional.empty()) options.mode = parse_int(positional[0], "mode");
gemm_variant(options.mode);
if (positional.size() > 1) options.m = parse_unsigned(positional[1], "M");
if (positional.size() > 2) options.n = parse_unsigned(positional[2], "N");
if (positional.size() > 3) options.k = parse_unsigned(positional[3], "K");
```

这里只覆盖用户提供的前若干项，其余成员继续保留 Options 中的默认值。各处的数量判断同时保证 `positional[index]` 不越界。

`gemm_variant(options.mode);` 虽然没有保存返回值，却有检查作用：当前实现找到合法模式就返回描述，找不到就抛出异常。这里复用描述查询函数验证 mode，避免在解析器里重复维护一套合法编号。

合法编号与是否允许实际执行属于不同检查：解析器检查 mode 是否存在；run 再检查变体的 implemented 标记等执行条件。

后续约束可以分为：

| 检查 | 负责的条件 |
| --- | --- |
| 三次 matrix_elements 调用 | A、B、C 形状及相应字节数可表示 |
| `options.iterations < 1` | 计时迭代次数必须为正 |
| `options.profile && options.check_only` | 两种运行请求不能同时开启 |

两个开关单独都合法，但组合可能冲突；因此，单值解析与整体配置检查需要不同层次。

### 2.3.6 按值返回配置，不返回局部引用

```cpp
return options;
```

函数返回类型是 Options，调用者得到一个结果对象，其有效性不依赖函数内局部变量的生存期。不能为了避免复制，改成返回指向这个局部对象的引用或指针。

对于这里的 `return options`，编译器可以实施具名返回值优化（NRVO），直接在结果位置构造对象；该优化允许发生，但不保证一定发生。即使没有进行优化，按值返回当前配置也具有正确的对象语义。[标准草案：拷贝消除](https://eel.is/c++draft/class.copy.elision)

这也允许入口写成 `run(parse_options(argc, argv))`：返回的临时结果可绑定到 run 的 const 引用参数，在这次调用期间有效。

### 2.3.7 当前解析器的明确边界

| 输入情况 | 当前源码行为 |
| --- | --- |
| 多次提供 `--warmup` | 后面的值覆盖前面的值 |
| `--warmup=2` | 不支持等号形式，被当成未知长选项 |
| `--warmup --plan` | 将 `--plan` 当作 warmup 的值，随后因非数字而报错 |
| 未知短选项 `-x` | 进入位置参数处理，随后由数值解析拒绝 |
| 单独的 `--` | 没有“结束选项解析”的特殊含义，会被拒绝 |
| `--help` 与其他错误参数并存 | 仍要先通过 parse_options，帮助标志不会跳过检查 |

这些是当前小型解析器的取舍，不应将它描述为支持所有常见命令行约定的通用库。

## 2.4 print_help：相邻字符串字面量的拼接

```cpp
std::puts(
    "Usage: sgemm ...\n"
    "Options:\n"
    "  --help  Show help\n");
```

这里相邻的普通字符串字面量会在翻译阶段合并成一个字符串；没有逗号，因此 puts 仍只接收一个参数。这种写法方便把长帮助文本按行排版，不需要运行时通过 std::string 的 `+` 拼接。[标准草案：字符串字面量](https://eel.is/c++draft/lex.string)

源码换行本身不会成为输出换行，文本里的 `\n` 才负责换行；puts 还会在输出末尾额外写入一个换行。字面量相邻拼接只适用于这种语法，两个字符串变量不能直接并排书写来拼接。

本节待自测：若收到 `sgemm 2 --warmup 0 64 --iterations 5 32 16`，positional 最终有哪些字符串？哪些 Options 成员被覆盖？暂不要求运行 GPU，先按读取游标推导。

# 3. validate：数值判定与错误诊断

> **作用**：比较已经位于主机内存的 GPU 输出与 CPU double 参考结果，统计误差并返回是否通过。数据拷贝与后续流程由调用者负责。

## 3.1 接口：借用输入，返回判定

```cpp
bool validate(const std::vector<float>& actual, const std::vector<double>& reference,
              const Options& options, const char* phase)
```

actual、reference 和 options 都通过 const 引用读取，沿用前文的借用方式。phase 是本次校验的阶段标签，例如 `"before"`、`"after"`，用于让日志说明结果来自哪次检查。

这里有三类输出：bool 返回是否通过；汇总日志报告错误规模和最严重程度；首个错误日志提供定位线索。数值不匹配时返回 false，由 run 决定是否停止实验并返回校验失败退出码。

这种分工让同一个校验函数可以在实验的不同阶段复用。

## 3.2 绝对误差与混合容差

```cpp
const double error = std::isfinite(actual[i])
    ? std::abs(static_cast<double>(actual[i]) - reference[i])
    : std::numeric_limits<double>::infinity();
const double allowed = options.atol + options.rtol * std::abs(reference[i]);
```

对有限的实际结果，单元素通过条件为：

```text
error = |actual - reference|
allowed = atol + rtol × |reference|
error ≤ allowed
```

atol 是绝对容差，提供接近零时仍可使用的误差额度；rtol 是相对容差，使允许误差随参考值的量级增大。当前公式将两者相加。

以默认 atol=rtol=0.0001 为例，下面按公式计算：

| reference | actual | error | allowed | 判定 |
| --- | --- | --- | --- | --- |
| 10000 | 10000.5 | 0.5 | 1.0001 | 通过 |
| 1 | 1.5 | 0.5 | 0.0002 | 失败 |
| 0 | 0 | 0 | 0.0001 | 通过 |

相同的绝对误差，在不同量级上可能得到不同判定。参考值为零时，相对容差项也为零，绝对容差继续提供误差额度。

浮点累加的顺序和精度会影响结果，CPU 与 GPU 也可能采用不同的运算组织方式，因此需要根据算法需求选择合适的容差。[NVIDIA：参考结果比较](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#reference-comparison)、[浮点数精度](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#numerical-accuracy-and-precision)

把实际值提升为 double，是在 double 精度下计算对照误差；它无法恢复 GPU float 计算已经舍入掉的信息。CPU double 结果提高了对照计算的精度，也仍属于有限精度计算。

## 3.3 条件运算符：构造诊断值

```cpp
条件 ? 条件为真时的表达式 : 条件为假时的表达式
```

`?:` 是条件运算符，会产生一个值，只有选中的分支会被求值。因此，它适合在声明 const 变量时，根据条件选择初始化值。[标准草案：条件运算符](https://eel.is/c++draft/expr.cond)

在 error 的初始化中，有限 actual 使用误差计算；非有限 actual 统一映射成正无穷，作为失败诊断值。`std::numeric_limits<double>::infinity()` 取得浮点类型的正无穷值。

为什么要显式处理 NaN？NaN 参与这里的 `>` 等有序比较时结果为 false，若只靠 `error > allowed`，可能漏掉非法结果。因此，后面的失败条件也显式包含 `!std::isfinite(actual[i])`。

## 3.4 scaled：误差占允许额度的多少倍

```cpp
const double scaled = !std::isfinite(error) ? std::numeric_limits<double>::infinity() :
    (allowed > 0 ? error / allowed :
     (error == 0 ? 0 : std::numeric_limits<double>::infinity()));
```

通常情况下，`scaled = error / allowed`。它是一个以容差为基准的无量纲比值：0.5 表示用掉一半允许额度，1 表示恰好到达边界，2 表示误差达到允许值的两倍。

这个指标用到了 atol 和 rtol，不能把它简单记为“相对误差”。

嵌套表达式可以按下面的分支理解。以下为讲解用等价展开，未修改源码；inf 代表正无穷：

```cpp
const double inf = std::numeric_limits<double>::infinity();
double scaled;
if (!std::isfinite(error))
    scaled = inf;
else if (allowed > 0)
    scaled = error / allowed;
else if (error == 0)
    scaled = 0;
else
    scaled = inf;
```

允许 atol 和 rtol 同时为零，就需要明确规定零容差下的诊断行为：误差为零时记为 0；误差大于零时记为无穷。条件分支避免直接计算 `0 / 0`。

**设计思维**：除了定义通过条件，还应为异常值和边界输入定义可解释的诊断结果。

## 3.5 一次遍历，收集不同层次的信息

```cpp
max_absolute = std::max(max_absolute, error);
max_scaled = std::max(max_scaled, scaled);
if (!std::isfinite(actual[i]) || error > allowed)
{
    if (failed == 0) first_bad = i;
    ++failed;
}
```

| 变量 | 回答的问题 |
| --- | --- |
| failed | 一共有多少元素不合格？ |
| max_absolute | 最大绝对误差是多少？ |
| max_scaled | 最严重的误差达到容差额度的多少倍？ |
| first_bad | 按遍历顺序，第一个错误在哪里？ |

两个最大值可能来自不同元素；first_bad 也不一定对应最大误差。它们分别服务于误差概况和快速定位。

`if (failed == 0)` 只在首次失败时记录下标，随后递增失败计数。循环继续扫描所有元素，才能获得完整数量和最大误差，而不是在首次失败后立即结束。

first_bad 初始为 0 并不表示“没有错误”；是否存在错误由 failed 决定。否则，下标 0 真正出错时就容易与“没找到”混淆。

这里的比较扫描时间复杂度为 O(M×N)，额外统计空间为 O(1)；构造 CPU GEMM 参考结果的 O(M×N×K) 开销属于 run 中的另一个步骤。

## 3.6 从线性下标定位到矩阵坐标

```cpp
first_bad / options.n  // row
first_bad % options.n  // col
```

对于行主序 C[M,N]：`i = row * N + col`，因此除以行宽 N 得到行号，取余得到列号。

例如 N=4、first_bad=6，对应零起始坐标 row=1、col=2。这让诊断信息可以直接关联到核函数的输出索引。

## 3.7 日志格式也属于工程接口

```text
Check | phase=... | status=... | checked=full | elements=... | failed=...
Mismatch | row=... | col=... | expected=... | actual=...
```

稳定的记录类型和 `key=value` 字段，同时方便人阅读和脚本提取。成功时输出汇总，失败时补充首个错误详情，避免一次打印整个错误矩阵。

源码中的格式说明符可以这样读：

| 格式 | 实参类型与含义 |
| --- | --- |
| `%s` | C 字符串，例如 phase 或 PASS/FAIL |
| `%zu` | size_t，例如元素数和下标 |
| `%.9g` | double，按 g 格式输出，精度为 9 位有效数字 |
| `%.12g` | double，精度为 12 位有效数字，用于详细对照 |

`g` 根据数值选择普通或科学计数形式，精度表示有效数字位数，并非固定小数位数；默认还会去掉不必要的尾随零。printf 的格式说明与实参类型需要匹配，不能因为某些类型在当前平台同宽就随意混用。[Microsoft：printf 格式说明](https://learn.microsoft.com/en-us/cpp/c-runtime-library/format-specification-syntax-printf-and-wprintf-functions?view=msvc-170)

## 3.8 当前函数依赖的输入契约

调用者需要保证两份数组形状匹配、长度为 M×N、N>0，并提供有限的参考结果及合理的有限非负容差。当前 run 按相同 count_c 分配 actual 和 reference，再产生参考数据。

validate 本身没有重新检查数组长度或 reference 的有限性。若将它提取到其他工程中复用，需要重新决定这些前提由调用者保证，还是由函数自行检查。

PASS 表示本次输入在指定容差下通过了完整输出比较，不等于证明所有输入都正确，也不替代越界、竞争和同步依赖方面的检查。

本节待自测：reference=0、atol=0、rtol=0 时，actual=0 与 actual=0.125 各自得到怎样的 error、allowed、scaled 和判定？先按分支推导，不要求运行 GPU。

# 4. run：把资源、算子与测量组织成一次实验

run 的主要职责是实验编排。先处理信息查询与执行条件，再准备数据和资源，最后选择检查、计时或采集路径。

```text
help / list / plan → 输出信息后返回
    ↓ 需要实际执行时
检查实现状态与设备限制 → 准备输入和 CPU 参考结果 → 分配并初始化显存
    ↓
首次 GEMM → 等待完成 → 校验 before
    ├─ check-only：返回
    ├─ profile：预热 → 开启采集 → GEMM → 等待完成 → 关闭采集 → 校验 after
    └─ 普通计时：预热 → 事件计时下重复 GEMM → 校验 after → 输出平均耗时和 GFLOP/s
```

## 4.1 用接口区分描述、计划与执行

```cpp
const auto& variant = gemm_variant(options.mode);
const auto launch = make_gemm_launch(options.mode, options.m, options.n);
```

variant 借用已有的 GemmVariant 描述，launch 则持有按当前形状计算得到的 GemmLaunch 值。这里的引用或值选择与被调用函数的返回接口相对应，不必把所有 auto 都写成引用。

将元数据、启动配置与真正执行拆开后，`--plan` 可以只计算并展示 grid、block 和 shared 大小；实际执行前再检查设备的线程数、网格大小和 shared 容量限制。昂贵的参考计算和资源分配安排在这些检查之后。

其中 `cudaDeviceProp properties{};` 用空花括号把这个结构体初始化到零等初始状态，随后由 cudaGetDeviceProperties 填充。

## 4.2 准备可复现的输入与独立参考

输入使用固定公式生成，A、B 的公式不同，数值包含正负号，并乘以 `0.03125f`，即可以用二进制精确表示的 1/32。这样可以复现实验输入，也避免统一常量输入掩盖部分索引错误。

CPU 参考计算采用 row → inner → col 的顺序：最内层 col 连续访问 B 的一行与 reference 的一行，提高局部性；double 用于参考累加。这一过程位于事件计时区间外。

run 通过 `CudaBuffer<float>` 持有显存，通过 `.get()` 借用设备指针交给 CUDA API。主机 vector 的 `.data()` 同样提供其连续存储的指针。这些调用提供底层接口所需的地址，资源所有权仍属于对应容器或封装对象，具体 RAII 实现留到头文件部分。

## 4.3 lambda：把已经绑定上下文的操作交给计时器

```cpp
auto gemm = [&]() {
    launch_gemm(options.mode, device_a.get(), device_b.get(), device_c.get(),
                options.m, options.n, options.k, launch);
};
auto check_result = [&](const char* phase) {
    CUDA_CHECK(cudaMemcpy(actual.data(), device_c.get(), count_c * sizeof(float), cudaMemcpyDeviceToHost));
    return validate(actual, reference, options, phase);
};
```

lambda 表达式产生一个具有编译器生成类型的闭包对象，auto 接住这个对象；`gemm()` 则调用它。[标准草案：闭包类型](https://eel.is/c++draft/expr.prim.lambda.closure)

`[&]` 使这里需要捕获的外部局部变量按引用捕获；lambda 可以使用现有配置和缓冲区，无需把它们重新列为每次调用的参数。check_result 的 phase 是每次调用时传入的参数，它的 bool 返回类型由 return 表达式推导。[标准草案：lambda 捕获](https://eel.is/c++draft/expr.prim.lambda.capture)

这里选择引用捕获也与所有权设计一致：CudaBuffer 禁止复制，直接将 gemm 改成 `[=]` 会尝试复制捕获的缓冲区对象，无法编译。引用捕获借用资源，不转移所有权，也不延长被捕获对象的生命周期；当前两个 lambda 都只在 run 内使用。

```cpp
const float ms = timer.benchmark(gemm, options.warmup, options.iterations);
```

这个接口把职责拆开：run 定义“一次工作是什么”，计时器负责“预热几次、执行几次、怎样测量”。捕获型 lambda 携带上下文，计时器通过模板接收可调用对象，具体模板和引用语义在 cuda_timer.cuh 一节继续展开。

当前 GEMM 接口要求每次覆盖 C，结果只依赖 A、B。重复调用因此使用相同工作负载，不会因为持续累加旧 C 而改变数值含义。

## 4.4 输出哨兵：使漏写能够显现

```cpp
CUDA_CHECK(cudaMemset(device_c.get(), 0xff, count_c * sizeof(float)));
```

cudaMemset 按字节填充；每个 float 的四个字节都写成 0xff，形成位模式 0xffffffff，在 CUDA 使用的 IEEE 754 binary32 表示中是 NaN。它不是把每个 float 赋成数值 255。[NVIDIA：cudaMemset](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html)

如果核函数遗漏某个输出，保留下来的 NaN 会被 validate 拒绝。若预先清零，遗漏位置恰好应为零时可能误通过。这体现了主动设计测试初始状态的思路。

当前代码只在首次执行前填充一次哨兵；后续 after 校验检查最终结果，不单独证明每一轮都完整写过输出。

## 4.5 将正确性检查放在测量边界两侧

首次执行后先同步并检查 before，避免对当前输入下结果不合格的实现继续报告性能；普通计时和 profile 路径结束后再检查 after。

| 路径 | 主要用途 | before 通过后的操作 |
| --- | --- | --- |
| check-only | 快速进行数值检查 | 直接返回 |
| 普通计时 | 获得重复执行的平均时间 | 预热、计时、检查 after、计算吞吐率 |
| profile | 采集核函数指标 | 预热、标记采集范围、检查 after |

gemm 调用完成只表示主机端的启动操作返回，GPU 核函数可能尚未完成。当前代码在读取首次结果和关闭采集范围前使用 cudaDeviceSynchronize，在事件计时路径中由计时器等待 stop 事件完成。这些同步让资源使用、读回结果与测量边界对齐。

### 普通计时的口径

当前计时器执行：预热 → 同步 → 记录 start → 重复调用 → 记录 stop 并等待 → 总时间除以迭代次数。CPU 参考计算、H2D/D2H 拷贝、结果校验和日志均位于事件区间外。[NVIDIA：CUDA 事件](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html)

该值表示当前提交方式和执行条件下的事件区间平均耗时；尤其对很短的核函数，CPU 提交造成的 GPU 空闲间隙也可能反映到区间中。它不等于整个程序的端到端耗时。

### profile 的采集范围

仓库 run_ncu.ps1 使用 `--profile-from-start off`，与 cudaProfilerStart / cudaProfilerStop 配合，将预热和前置检查放在采集范围外。范围内提供一次用于采集的逻辑 GEMM 启动；NCU 为收集指标可能重放该核函数。程序中的标记需由外部 profiler 配合才形成采集报告。[NVIDIA：Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)

## 4.6 吞吐率：明确计算量与时间单位

代码按 GEMM 的常用计数约定使用约 `2×M×N×K` FLOP：一次乘加计两次浮点运算。ms 是平均毫秒数，因此：

```text
GFLOP/s = FLOP / (ms × 10^-3) / 10^9
        = FLOP / (ms × 10^6)
```

这解释了 `flops / (static_cast<double>(ms) * 1e6)` 的单位换算。结果在 after 校验通过、时间有限且为正后才输出。

本节待思考：若把 cudaMemcpy 与 validate 一起移进传给 benchmark 的 lambda，得到的时间还对应现在的工作负载吗？重点说明测量范围，而不是背语法。

# PowerShell 部分

## 1. 先读懂一条实验命令

PowerShell 是解释并执行命令的 shell，也是一门脚本语言。终端是输入命令、显示输出的界面；`.ps1` 文件则保存 PowerShell 代码。

本节先掌握命令调用规则，脚本内部的管道、重定向和参数展开后续再讲。

### 1.1 先区分两层参数

```text
PowerShell
    ├─ 直接启动 sgemm.exe → 程序内 parse_options 解释参数
    └─ 运行 run_benchmark.ps1 → PowerShell 绑定脚本参数
                                  ↓
                             脚本逐次启动 sgemm.exe
```

| 接收者 | 参数示例 | 参数规则来自哪里 |
| --- | --- | --- |
| PowerShell 命令，例如 Set-Location | `-Path` | 该命令声明的参数 |
| PowerShell 脚本 run_benchmark.ps1 | `-Modes`、`-Shapes`、`-CheckOnly` | 脚本开头的 param 声明 |
| 外部程序 sgemm.exe | `0 127 193 65 --plan` | 程序自己的命令行解析代码 |

单横线和双横线不是统一的功能分类。要先识别接收参数的是谁，再按它的接口填写；不能把脚本的 `-CheckOnly` 直接替换成程序的 `--check-only`，反之亦然。

### 1.2 命令结构：命令名后用空格放参数

```powershell
.\build\labs\sgemm\Release\sgemm.exe 0 127 193 65 --plan
```

可以拆成：

| 部分 | 含义 |
| --- | --- |
| `.\build\labs\sgemm\Release\sgemm.exe` | 要执行的程序路径 |
| `0` | 位置参数 mode |
| `127 193 65` | 依次为 M、N、K |
| `--plan` | 由 sgemm.exe 解释的选项 |

PowerShell 的常见命令调用形式是 `命令 参数 参数`。需要计算一个参数值时，再用括号明确表达式：

```powershell
Write-Output 2+3     # 输出文本 2+3
Write-Output (2+3)   # 输出数值 5
```

命令参数模式与算术表达式模式有不同规则，不能把 C++ 的函数调用形式直接套在命令上。每行通常是一条命令，不必写分号；`#` 开始行注释。[Microsoft：PowerShell 解析规则](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_parsing?view=powershell-7.6)

### 1.3 路径从哪里开始算

```powershell
Get-Location
Set-Location 'D:\CG\Kernel_Optimization\cuda-labs'
```

Get-Location 查看当前位置；Set-Location 切换目录，常见别名是 cd。

| 路径写法 | 含义 |
| --- | --- |
| `.` | 当前目录 |
| `..` | 上一级目录 |
| `.\labs\sgemm\...` | 从当前目录往下找 |
| `D:\CG\Kernel_Optimization\cuda-labs\...` | 绝对路径 |

当前目录取决于终端的位置，不会因为正在编辑某个脚本就自动变成那个脚本所在目录。

PowerShell 通常不会仅因某程序位于当前目录就按裸名称找到它；显式使用 `.\程序名.exe` 或完整路径可以明确执行目标。[Microsoft：命令查找顺序](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_command_precedence?view=powershell-7.6)

### 1.4 命名参数、开关与数组

下面这条命令在仓库根目录运行时，会由脚本自动构建并检查所选配置。本节只分析调用含义，未执行该脚本：

```powershell
.\labs\sgemm\scripts\run_benchmark.ps1 -Modes 0,2 -Shapes '3x5x7','17x19x23' -Repeat 1 -CheckOnly
```

| 片段 | 含义 |
| --- | --- |
| `-Modes 0,2` | Modes 接收由两个整数组成的数组 |
| `-Shapes '3x5x7','17x19x23'` | Shapes 接收由两个形状字符串组成的数组 |
| `-Repeat 1` | 每种配置运行一轮 |
| `-CheckOnly` | 开启只校验、不计时的开关 |

类型对应脚本开头的 `[int[]]$Modes`、`[string[]]$Shapes`、`[int]$Repeat` 和 `[switch]$CheckOnly`。方括号里的类型决定参数接收什么，`[]` 表示数组。

在这条 PowerShell 脚本调用中，逗号将多个值组成一个数组，交给同一个参数；它不负责分隔不同参数。因此，`-Modes 0,2` 与 `-Modes '0,2'` 含义不同，后者是一个包含逗号的字符串。[Microsoft：命令参数](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_parameters?view=powershell-7.6)

两个 mode 乘两个 shape，构成四种配置，Repeat=1 表示每种配置跑一轮。Repeat 控制脚本轮次；Iterations 控制每个程序进程内普通计时的重复启动次数，CheckOnly 路径不进行普通计时。

逗号的规则还取决于接收者。直接运行原生 exe 时，不能用 `0,2` 要求本项目一次执行两个 mode；脚本正是通过循环逐次启动程序实现这种批量工作。NCU 的 `-Metrics '指标一,指标二'` 则刻意接收一个含逗号的字符串，再由脚本拆分。

### 1.5 单引号、双引号与变量

```powershell
$taskSize = 128
'M=$taskSize'     # 输出 M=$taskSize
"M=$taskSize"     # 输出 M=128
```

变量名以 `$` 开头。单引号保留文字，双引号会展开变量；不需要变量展开时，固定文本和路径可以使用单引号。PowerShell 的转义字符是反引号，Windows 路径里的反斜杠通常直接写即可。[Microsoft：引号规则](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_quoting_rules?view=powershell-7.6)

引号还能将含空格的内容保持为一个参数，例如：

```powershell
.\labs\sgemm\scripts\run_benchmark.ps1 -BuildDir 'build debug' -Modes 0 -CheckOnly
```

这一例中 BuildDir 收到的目录名是完整的 `build debug`。

### 1.6 &：调用由字符串或变量指定的命令

```powershell
$taskExe = '.\build\labs\sgemm\Release\sgemm.exe'
& $taskExe 0 127 193 65 --plan
```

当命令路径存放在变量中，使用调用运算符 `&` 执行它。带引号的路径同理：

```powershell
& '.\build\labs\sgemm\Release\sgemm.exe' 0 127 193 65 --plan
```

单独写一个带引号的路径只是产生字符串；`&` 才要求调用该命令。这里的 & 是前缀调用运算符，不是 C++ 的取地址符号。[Microsoft：调用运算符](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_operators?view=powershell-7.6)

命令路径与参数应分开存放，不能期待 `&` 把一个包含整条命令的字符串重新解析成路径和参数。后续脚本中的参数数组正用于清楚地保留各参数边界。

### 1.7 长命令换行：读懂反引号

```powershell
.\labs\sgemm\scripts\run_benchmark.ps1 `
    -Modes 0,2 `
    -Shapes '3x5x7','17x19x23' `
    -Repeat 1 `
    -CheckOnly
```

行末反引号表示命令延续到下一行，整段仍然是一条命令。反引号必须紧贴换行，后面不能再有空格或注释；这里的缩进只帮助阅读。

较长的可复用命令可以用参数展开改善排版，后续再讲。当前先能识别这种续行形式。[Microsoft：续行](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_parsing?view=powershell-7.6)

### 1.8 从哪里查参数

```powershell
Get-Help .\labs\sgemm\scripts\run_benchmark.ps1 -Parameter *
.\build\labs\sgemm\Release\sgemm.exe --help
```

Get-Help 可以查看 PowerShell 脚本声明的参数；sgemm.exe 则使用自身实现的 --help。文档中的 `[mode=0]`、`<参数值>` 等说明性占位符，不需要原样输入终端；代码块的语言标签 powershell 和提示符 `PS D:\...>` 也不属于命令正文。

本节已在 PowerShell 7.6.5 中核对数组参数绑定、单双引号、括号表达式，并实际执行上述 --plan 示例。该计划显示 M=127、N=193、K=65，grid=(13,8)，未启动 GEMM。跑分脚本调用仅用于讲解。

本节待练习：从仓库根目录，写出“只检查 mode 1，形状为 17×19×23，运行一轮”的脚本命令。先写一行，再尝试用反引号拆成多行。
