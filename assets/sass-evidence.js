"use strict";

// Semantic stepping only: these are illustrative lane values, not a GPU trace.
document.querySelectorAll("[data-sass-stepper]").forEach((root) => {
    const initial = { R0: "0x0", R27: "0x0", R29: 17 };
    const steps = [
        { code: "/*03b0*/ LDS R28, [R27+0x400];", writes: { R28: 11 }, why: "从示例 shared 字节地址 0x400 读取 B[0,0]=11。R27 是地址基值，R28 接收数据。" },
        { code: "/*03c0*/ LDS.128 R8, [R0];", writes: { R8: 2, R9: 3, R10: 5, R11: 7 }, why: "从 0x0 起读取四个相邻 float：A[0,0..3]。一条宽加载写 R8～R11；.128 表示每个参与线程读取 128 bit。" },
        { code: "/*03d0*/ LDS R32, [R27+0x440];", writes: { R32: 13 }, why: "0x440−0x400=64 字节，对应 B 的下一行（16 个 float）。示例 B[1,0]=13。" },
        { code: "/*0420*/ FFMA R8, R8, R28, R29;", writes: { R8: 39 }, why: "先读旧 R8=2、R28=11、R29=17，再写新 R8=2×11+17=39。R8 从 A 操作数变成部分和。" },
        { code: "/*0440*/ FFMA R9, R9, R32, R8;", writes: { R9: 78 }, why: "读取旧 R9=3、R32=13、新 R8=39，再写 R9=3×13+39=78。后一条消费前一条产生的部分和。" },
    ];
    let position = 0;
    const next = root.querySelector("[data-step]");
    const reset = root.querySelector("[data-reset]");
    function render() {
        const registers = { ...initial };
        for (let i = 0; i < position; ++i) Object.assign(registers, steps[i].writes);
        const current = position ? steps[position - 1] : null;
        root.querySelector("[data-instruction]").textContent = current ? current.code : "起点：R0=0x0，R27=0x0，旧部分和 R29=17。";
        root.querySelector("[role=status]").textContent = current ? current.why : "先预测第一条指令会改哪个寄存器，再点下一步。中间不相关指令已省略；本组件不模拟调度、延迟或 bank。";
        root.querySelector("[data-registers]").textContent = Object.entries(registers).map(([key, value]) => `${key.padEnd(4)} = ${value}`).join("\n");
        next.disabled = position === steps.length;
        next.textContent = position === steps.length ? "已到片段末尾" : "执行下一步";
    }
    next.addEventListener("click", () => { position += 1; render(); });
    reset.addEventListener("click", () => { position = 0; render(); });
    render();
});
