document.addEventListener("DOMContentLoaded", () => {
    const blocks = Array.from(document.querySelectorAll(".semantic-highlight-block"));
    const hoverCard = document.getElementById("semanticHoverCard");
    const hoverUser = document.getElementById("semanticHoverUser");
    const hoverDoc = document.getElementById("semanticHoverDoc");
    const hoverScore = document.getElementById("semanticHoverScore");

    if (!blocks.length || !hoverCard || !hoverUser || !hoverDoc || !hoverScore) {
        return;
    }

    const parsePairList = (raw) =>
        String(raw || "")
            .split(",")
            .map((item) => item.trim())
            .filter(Boolean);

    const hasPairOverlap = (candidateIds, activeIds) => {
        if (!candidateIds.length || !activeIds.length) {
            return false;
        }
        const activeSet = new Set(activeIds);
        return candidateIds.some((id) => activeSet.has(id));
    };

    const clearBlockFocus = (block) => {
        block.querySelectorAll(".semantic-sentence").forEach((el) => {
            el.classList.remove("is-focus", "is-linked");
        });
        block.querySelectorAll(".semantic-pair-row").forEach((row) => {
            row.classList.remove("is-focus");
        });
    };

    const hideHoverCard = () => {
        hoverCard.hidden = true;
        hoverUser.textContent = "";
        hoverDoc.textContent = "";
        hoverScore.textContent = "";
    };

    const showHoverCardFromRow = (row) => {
        hoverUser.textContent = `Your sentence: ${row.dataset.userText || ""}`;
        hoverDoc.textContent = `Matched sentence: ${row.dataset.docText || ""}`;
        hoverScore.textContent = `Semantic score: ${row.dataset.score || "0"}%`;
        hoverCard.hidden = false;
    };

    const applyPairFocus = (block, pairIds, focusEl = null) => {
        clearBlockFocus(block);
        if (!pairIds.length) {
            hideHoverCard();
            return;
        }

        block.querySelectorAll(".semantic-sentence").forEach((sentenceEl) => {
            const sentencePairIds = parsePairList(sentenceEl.dataset.pairList);
            if (hasPairOverlap(sentencePairIds, pairIds)) {
                sentenceEl.classList.add("is-linked");
            }
        });

        const matchedRows = [];
        block.querySelectorAll(".semantic-pair-row").forEach((row) => {
            if (pairIds.includes(row.dataset.pairId || "")) {
                row.classList.add("is-focus");
                matchedRows.push(row);
            }
        });

        if (focusEl && focusEl.classList.contains("semantic-sentence")) {
            focusEl.classList.add("is-focus");
        }
        if (focusEl && focusEl.classList.contains("semantic-pair-row")) {
            focusEl.classList.add("is-focus");
        }

        if (matchedRows.length) {
            showHoverCardFromRow(matchedRows[0]);
        } else {
            hideHoverCard();
        }
    };

    blocks.forEach((block) => {
        const sentences = block.querySelectorAll(".semantic-sentence");
        const rows = block.querySelectorAll(".semantic-pair-row");

        sentences.forEach((sentenceEl) => {
            const ids = parsePairList(sentenceEl.dataset.pairList);
            if (!ids.length) {
                return;
            }
            sentenceEl.addEventListener("mouseenter", () => {
                applyPairFocus(block, ids, sentenceEl);
            });
        });

        rows.forEach((row) => {
            const pairId = row.dataset.pairId || "";
            if (!pairId) {
                return;
            }
            row.addEventListener("mouseenter", () => {
                applyPairFocus(block, [pairId], row);
            });
        });

        block.addEventListener("mouseleave", () => {
            clearBlockFocus(block);
            hideHoverCard();
        });
    });
});
