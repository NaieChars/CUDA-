"use strict";
document.querySelectorAll("[data-gemm-outer-product]").forEach((root) => {
    const a = JSON.parse(root.dataset.a), b = JSON.parse(root.dataset.b);
    const rows = a.length, cols = b[0].length, k = b.length;
    const step = root.querySelector("[data-step]");
    const status = root.querySelector("[role=status]");
    let p, acc;
    const table = (target, values) => {
        target.innerHTML = "<table><tbody>" + values.map(row =>
            "<tr>" + row.map(value => "<td>" + value + "</td>").join("") + "</tr>"
        ).join("") + "</tbody></table>";
    };
    const render = () => {
        const last = p - 1;
        table(root.querySelector("[data-fragment-a]"),
            Array.from({ length: rows }, (_, i) => [last < 0 ? "·" : a[i][last]]));
        table(root.querySelector("[data-fragment-b]"),
            [Array.from({ length: cols }, (_, j) => last < 0 ? "·" : b[last][j])]);
        table(root.querySelector("[data-acc]"), acc);
        step.disabled = p === k;
        step.textContent = p === k ? "所有 K 项已累加" : "读取下一组，并更新全部累加器";
    };
    const reset = () => {
        p = 0;
        acc = Array.from({ length: rows }, () => Array(cols).fill(0));
        status.textContent = "累加器从零开始；每次读取两组私有操作数，形成一次外积。";
        render();
    };
    step.addEventListener("click", () => {
        if (p === k) return;
        for (let i = 0; i < rows; ++i)
            for (let j = 0; j < cols; ++j)
                acc[i][j] += a[i][p] * b[p][j];
        status.textContent = "p=" + p + "：读取 " + (rows + cols)
            + " 个 shared 标量，完成 " + (rows * cols) + " 次乘加。"
            + "同一个 A 操作数沿列复用，同一个 B 操作数沿行复用。";
        p += 1;
        render();
    });
    root.querySelector("[data-reset]").addEventListener("click", reset);
    reset();
});
