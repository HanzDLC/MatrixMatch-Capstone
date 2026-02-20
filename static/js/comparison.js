document.addEventListener("DOMContentLoaded", () => {
    const keywordInput = document.getElementById("keywordInput");
    const keywordList = document.getElementById("keywordList");
    const keywordsHidden = document.getElementById("keywordsHidden");
    const keywordCountText = document.getElementById("keywordCountText");
    const form = document.getElementById("comparisonForm");
    const threshold = document.getElementById("threshold");
    const thresholdValue = document.getElementById("thresholdValue");
    const abstractField = document.getElementById("abstract");
    const abstractMeta = document.getElementById("abstractMeta");

    const keywords = [];

    const normalizeKeyword = (value) => value.trim().replace(/\s+/g, " ");

    const renderKeywords = () => {
        keywordList.innerHTML = "";
        keywords.forEach((keyword, index) => {
            const chip = document.createElement("span");
            chip.className = "chip";
            chip.textContent = keyword;

            const removeBtn = document.createElement("button");
            removeBtn.type = "button";
            removeBtn.className = "chip__remove";
            removeBtn.setAttribute("aria-label", `Remove ${keyword}`);
            removeBtn.textContent = "x";
            removeBtn.addEventListener("click", () => {
                keywords.splice(index, 1);
                renderKeywords();
            });

            chip.appendChild(removeBtn);
            keywordList.appendChild(chip);
        });

        keywordsHidden.value = JSON.stringify(keywords);
        keywordCountText.textContent = `${keywords.length} keyword${keywords.length !== 1 ? "s" : ""} added.`;
        keywordCountText.classList.toggle("is-warning", keywords.length < 5);
        keywordCountText.classList.toggle("is-good", keywords.length >= 5);
    };

    const addKeyword = () => {
        const value = normalizeKeyword(keywordInput.value);
        if (!value) {
            return;
        }
        const exists = keywords.some((entry) => entry.toLowerCase() === value.toLowerCase());
        if (!exists) {
            keywords.push(value);
            renderKeywords();
        }
        keywordInput.value = "";
    };

    if (keywordInput) {
        keywordInput.addEventListener("keydown", (event) => {
            if (event.key === "Enter" || event.key === ",") {
                event.preventDefault();
                addKeyword();
            }
        });

        keywordInput.addEventListener("blur", addKeyword);
    }

    renderKeywords();

    if (form) {
        form.addEventListener("submit", (event) => {
            if (keywords.length < 5) {
                event.preventDefault();
                window.alert("Please add at least 5 keywords before running comparison.");
            }
        });
    }

    if (threshold && thresholdValue) {
        const updateThresholdText = () => {
            thresholdValue.textContent = `${threshold.value}%`;
        };
        threshold.addEventListener("input", updateThresholdText);
        updateThresholdText();
    }

    if (abstractField && abstractMeta) {
        const updateAbstractMeta = () => {
            const text = abstractField.value || "";
            const words = text.trim() ? text.trim().split(/\s+/).length : 0;
            abstractMeta.textContent = `${words} words | ${text.length} characters`;
        };
        abstractField.addEventListener("input", updateAbstractMeta);
        updateAbstractMeta();
    }
});
