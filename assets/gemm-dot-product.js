"use strict";
document.querySelectorAll("[data-gemm-dot-product]").forEach((root) => {
    const a = JSON.parse(root.dataset.a);
    const b = JSON.parse(root.dataset.b);
    const m = a.length, k = a[0].length, n = b[0].length;
    if (b.length !== k || a.some((r) => r.length !== k) || b.some((r) => r.length !== n))
        throw new Error("Inconsistent demo matrix dimensions");
    const rowSelect = root.querySelector("[data-row]");
    const colSelect = root.querySelector("[data-col]");
    const stepButton = root.querySelector("[data-step]");
    const resetButton = root.querySelector("[data-reset]");
    const status = root.querySelector("[role=status]");
    let step = 0;
    function options(select, count) {
        for (let i = 0; i < count; ++i) {
            const option = document.createElement("option");
            option.value = String(i); option.textContent = String(i);
            select.appendChild(option);
        }
    }
    options(rowSelect, m); options(colSelect, n);
    rowSelect.value = String(Math.min(1, m - 1));
    colSelect.value = String(Math.min(2, n - 1));
    function matrix(target, rows, columns, value, highlighted) {
        target.replaceChildren();
        const table = document.createElement("table");
        const body = document.createElement("tbody");
        for (let r = 0; r < rows; ++r) {
            const tr = document.createElement("tr");
            for (let c = 0; c < columns; ++c) {
                const td = document.createElement("td");
                td.textContent = String(value(r, c));
                if (highlighted(r, c)) td.className = "matrix-active";
                tr.appendChild(td);
            }
            body.appendChild(tr);
        }
        table.appendChild(body); target.appendChild(table);
    }
    function render() {
        const row = Number(rowSelect.value), col = Number(colSelect.value);
        let sum = 0;
        for (let p = 0; p < step; ++p) sum += a[row][p] * b[p][col];
        matrix(root.querySelector("[data-matrix-a]"), m, k, (r, c) => a[r][c],
               (r, c) => r === row && (step === 0 || c === step - 1));
        matrix(root.querySelector("[data-matrix-b]"), k, n, (r, c) => b[r][c],
               (r, c) => c === col && (step === 0 || r === step - 1));
        matrix(root.querySelector("[data-matrix-c]"), m, n,
               (r, c) => r === row && c === col ? (step === k ? sum : "待写") : "·",
               (r, c) => r === row && c === col);
        status.textContent = step === 0 ? "目标 C[" + row + "," + col + "]；线程私有 sum=0，C 尚未写出。" :
            "p=" + (step - 1) + "：A[" + row + "," + (step - 1) + "] × B[" + (step - 1) + "," + col +
            "] = " + a[row][step - 1] + "×" + b[step - 1][col] + "；sum=" + sum +
            (step === k ? "，遍历 K 完成，写入目标 C。" : "，尚未写出 C。");
        stepButton.disabled = step === k;
    }
    stepButton.addEventListener("click", () => { if (step < k) { ++step; render(); } });
    resetButton.addEventListener("click", () => { step = 0; render(); });
    rowSelect.addEventListener("change", () => { step = 0; render(); });
    colSelect.addEventListener("change", () => { step = 0; render(); });
    render();
});
