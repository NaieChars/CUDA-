"use strict";
document.querySelectorAll("[data-gemm-tiling]").forEach((root) => {
    const a = JSON.parse(root.dataset.a);
    const b = JSON.parse(root.dataset.b);
    const t = Number(root.dataset.tile);
    const m = a.length, k = a[0].length, n = b[0].length;
    const rowSelect = root.querySelector("[data-block-row]");
    const colSelect = root.querySelector("[data-block-col]");
    const step = root.querySelector("[data-step]");
    const status = root.querySelector("[role=status]");
    const fill = (select, count) => {
        select.innerHTML = Array.from({ length: count }, (_, i) =>
            '<option value="' + i + '">' + i + '</option>').join("");
        select.value = "0";
    };
    fill(rowSelect, Math.ceil(m / t));
    fill(colSelect, Math.ceil(n / t));
    let k0, loaded, done, sums;
    const table = (target, cell) => {
        target.innerHTML = "<table><tbody>" + Array.from({ length: t }, (_, y) =>
            "<tr>" + Array.from({ length: t }, (_, x) => {
                const entry = cell(y, x);
                return '<td title="' + entry.title + '">' + entry.value + "</td>";
            }).join("") + "</tr>").join("") + "</tbody></table>";
    };
    const aCell = (y, x) => {
        const row = Number(rowSelect.value) * t + y, p = k0 + x;
        const valid = row < m && p < k;
        return { value: valid ? a[row][p] : "0*",
            title: valid ? "A[" + row + "," + p + "]" : "A 越界：写入零" };
    };
    const bCell = (y, x) => {
        const p = k0 + y, col = Number(colSelect.value) * t + x;
        const valid = p < k && col < n;
        return { value: valid ? b[p][col] : "0*",
            title: valid ? "B[" + p + "," + col + "]" : "B 越界：写入零" };
    };
    const render = () => {
        const empty = () => ({ value: "·", title: "尚未载入" });
        table(root.querySelector("[data-tile-a]"), loaded ? aCell : empty);
        table(root.querySelector("[data-tile-b]"), loaded ? bCell : empty);
        table(root.querySelector("[data-sums]"), (y, x) => {
            const row = Number(rowSelect.value) * t + y;
            const col = Number(colSelect.value) * t + x;
            const valid = row < m && col < n;
            return { value: valid ? sums[y][x] : "×",
                title: valid ? "线程私有 sum；最终才写 C" : "输出越界，仍参与加载和同步" };
        });
        step.textContent = done ? "全部 K 段已完成" : loaded
            ? "累加本段，并经过第二道屏障" : "载入下一段，并经过第一道屏障";
        step.disabled = done;
    };
    const reset = () => {
        k0 = -t; loaded = false; done = false;
        sums = Array.from({ length: t }, () => Array(t).fill(0));
        status.textContent = "每个线程的 sum=0。先协作载入第一对 tile。";
        render();
    };
    step.addEventListener("click", () => {
        if (done) return;
        if (!loaded) {
            k0 += t; loaded = true;
            status.textContent = "k0=" + k0 + "：两块数据已载入，第一道屏障完成，可以读取。";
        } else {
            for (let y = 0; y < t; ++y)
                for (let x = 0; x < t; ++x)
                    for (let e = 0; e < t; ++e) {
                        const av = aCell(y, e).value, bv = bCell(e, x).value;
                        sums[y][x] += (av === "0*" ? 0 : av) * (bv === "0*" ? 0 : bv);
                    }
            done = k0 + t >= k;
            status.textContent = "k0=" + k0 + "：本段已加入私有 sum，第二道屏障完成。"
                + (done ? "有效输出可以写回 C。" : "下一步可覆盖 shared，sum 继续保留。");
            // Keep the just-used tiles visible until the next cooperative load.
            render();
            if (!done) {
                loaded = false;
                step.textContent = "载入下一段，并经过第一道屏障";
            }
            return;
        }
        render();
    });
    rowSelect.addEventListener("change", reset);
    colSelect.addEventListener("change", reset);
    root.querySelector("[data-reset]").addEventListener("click", reset);
    reset();
});
