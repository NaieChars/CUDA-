"use strict";
document.querySelectorAll("[data-warp-reduction]").forEach((root) => {
    const initial = Array.from({length: 32}, (_, i) => i + 1);
    const offsets = [16, 8, 4, 2, 1];
    const states = [initial];
    offsets.forEach((offset) => {
        const previous = states[states.length - 1];
        states.push(previous.map((v, lane) => v + previous[lane + offset < 32 ? lane + offset : lane]));
    });
    let step = 0;
    const grid = root.querySelector("[data-lanes]");
    const status = root.querySelector("[role=status]");
    const previousButton = root.querySelector("[data-previous]");
    const nextButton = root.querySelector("[data-next]");
    function render() {
        const meaningful = step === 0 ? 32 : offsets[step - 1];
        grid.replaceChildren();
        states[step].forEach((value, lane) => {
            const cell = document.createElement("div");
            cell.className = "lane-cell";
            cell.textContent = "lane " + lane + ": " + (lane < meaningful ? value : "·");
            cell.setAttribute("aria-label", lane < meaningful ? "lane " + lane + " 当前有效值 " + value : "lane " + lane + " 不属于本轮后续有效前缀");
            grid.appendChild(cell);
        });
        status.textContent = step === 0 ? "初始输入：lane i 持有 i+1。" : "offset=" + offsets[step - 1] + " 完成；lane 0 的值为 " + states[step][0] + "。";
        previousButton.disabled = step === 0;
        nextButton.disabled = step === offsets.length;
    }
    previousButton.addEventListener("click", () => { if (step > 0) { step--; render(); } });
    nextButton.addEventListener("click", () => { if (step < offsets.length) { step++; render(); } });
    render();
});
