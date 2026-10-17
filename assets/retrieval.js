"use strict";
document.querySelectorAll("form[data-retrieval]").forEach((form) => {
    form.addEventListener("submit", (event) => {
        event.preventDefault();
        const fields = Array.from(form.querySelectorAll("input[data-answer]"));
        const correct = fields.every((field) => field.value.trim() !== "" && Number(field.value) === Number(field.dataset.answer));
        form.querySelector("[role=status]").textContent = correct ? form.dataset.correct : form.dataset.retry;
    });
});
