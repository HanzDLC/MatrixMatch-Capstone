document.addEventListener("DOMContentLoaded", () => {
    const searchInput = document.getElementById("researcherSearch");
    const table = document.getElementById("researchersTable");
    const sortBtn = document.getElementById("sortResearchersBtn");
    const filterBtn = document.getElementById("filterResearchersBtn");
    const exportBtn = document.getElementById("exportResearchersBtn");
    let showRecentOnly = false;

    const getRows = () => (table ? Array.from(table.querySelectorAll("tbody tr")) : []);
    const isRecentRow = (row) => {
        const value = row.dataset.registered;
        if (!value) {
            return false;
        }
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        const cutoff = new Date(today);
        cutoff.setDate(today.getDate() - 30);
        const dateValue = new Date(value);
        return !Number.isNaN(dateValue.getTime()) && dateValue >= cutoff;
    };

    const applyTableFilters = () => {
        if (!table) {
            return;
        }
        const term = (searchInput?.value || "").toLowerCase().trim();
        getRows().forEach((row) => {
            const text = row.innerText.toLowerCase();
            const passesSearch = text.includes(term);
            const passesDate = !showRecentOnly || isRecentRow(row);
            row.style.display = passesSearch && passesDate ? "" : "none";
        });
    };

    if (table && searchInput) {
        searchInput.addEventListener("input", () => {
            applyTableFilters();
        });
    }

    if (table && sortBtn) {
        let asc = false;
        sortBtn.addEventListener("click", () => {
            const tbody = table.querySelector("tbody");
            const rows = Array.from(tbody.querySelectorAll("tr"));
            rows.sort((a, b) => {
                const aDate = a.dataset.registered || "";
                const bDate = b.dataset.registered || "";
                return asc ? aDate.localeCompare(bDate) : bDate.localeCompare(aDate);
            });
            rows.forEach((row) => tbody.appendChild(row));
            asc = !asc;
            sortBtn.textContent = asc ? "Sort (Asc)" : "Sort (Desc)";
        });
    }

    if (table && filterBtn) {
        filterBtn.addEventListener("click", () => {
            showRecentOnly = !showRecentOnly;
            filterBtn.textContent = showRecentOnly ? "Filters (30d)" : "Filters";
            applyTableFilters();
        });
    }

    document.querySelectorAll(".js-delete-researcher-form").forEach((form) => {
        form.addEventListener("submit", (event) => {
            const confirmDelete = window.confirm(
                "Delete this researcher and all comparison history for this account?"
            );
            if (!confirmDelete) {
                event.preventDefault();
            }
        });
    });

    if (table && exportBtn) {
        exportBtn.addEventListener("click", () => {
            const headers = Array.from(table.querySelectorAll("thead th"))
                .slice(0, 4)
                .map((th) => th.textContent.trim());
            const rows = Array.from(table.querySelectorAll("tbody tr")).map((row) =>
                Array.from(row.querySelectorAll("td"))
                    .slice(0, 4)
                    .map((cell) => `"${cell.innerText.replace(/\s+/g, " ").trim().replace(/"/g, '""')}"`)
                    .join(",")
            );

            const csv = [headers.join(","), ...rows].join("\n");
            const blob = new Blob([csv], {type: "text/csv;charset=utf-8;"});
            const url = URL.createObjectURL(blob);
            const anchor = document.createElement("a");
            anchor.href = url;
            anchor.download = "researchers.csv";
            document.body.appendChild(anchor);
            anchor.click();
            anchor.remove();
            URL.revokeObjectURL(url);
        });
    }

    applyTableFilters();

    const canvas = document.getElementById("activityChart");
    const rawDates = Array.isArray(window.matrixmatchResearcherDates)
        ? window.matrixmatchResearcherDates
        : [];
    if (!canvas || !rawDates.length) {
        return;
    }

    const context = canvas.getContext("2d");
    if (!context) {
        return;
    }

    const today = new Date();
    today.setHours(0, 0, 0, 0);

    const dayKeys = [];
    for (let offset = 6; offset >= 0; offset -= 1) {
        const date = new Date(today);
        date.setDate(today.getDate() - offset);
        dayKeys.push(date.toISOString().slice(0, 10));
    }

    const countMap = Object.fromEntries(dayKeys.map((key) => [key, 0]));
    rawDates.forEach((value) => {
        if (!value) {
            return;
        }
        if (countMap[value] !== undefined) {
            countMap[value] += 1;
        }
    });

    const values = dayKeys.map((key) => countMap[key]);
    const width = canvas.width;
    const height = canvas.height;
    const leftPadding = 44;
    const rightPadding = 20;
    const topPadding = 20;
    const bottomPadding = 38;
    const chartWidth = width - leftPadding - rightPadding;
    const chartHeight = height - topPadding - bottomPadding;
    const maxValue = Math.max(...values, 1);

    context.clearRect(0, 0, width, height);

    context.strokeStyle = "rgba(130, 149, 176, 0.28)";
    context.lineWidth = 1;
    for (let i = 0; i <= 4; i += 1) {
        const y = topPadding + (chartHeight / 4) * i;
        context.beginPath();
        context.moveTo(leftPadding, y);
        context.lineTo(width - rightPadding, y);
        context.stroke();
    }

    const points = values.map((value, index) => {
        const x = leftPadding + (chartWidth / (values.length - 1)) * index;
        const y = topPadding + chartHeight - (value / maxValue) * chartHeight;
        return {x, y, value};
    });

    const fillGradient = context.createLinearGradient(0, topPadding, 0, topPadding + chartHeight);
    fillGradient.addColorStop(0, "rgba(30, 166, 246, 0.34)");
    fillGradient.addColorStop(1, "rgba(30, 166, 246, 0.03)");

    context.beginPath();
    context.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i += 1) {
        const prev = points[i - 1];
        const point = points[i];
        const controlX = (prev.x + point.x) / 2;
        context.quadraticCurveTo(controlX, prev.y, point.x, point.y);
    }
    context.lineTo(points[points.length - 1].x, topPadding + chartHeight);
    context.lineTo(points[0].x, topPadding + chartHeight);
    context.closePath();
    context.fillStyle = fillGradient;
    context.fill();

    context.beginPath();
    context.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i += 1) {
        const prev = points[i - 1];
        const point = points[i];
        const controlX = (prev.x + point.x) / 2;
        context.quadraticCurveTo(controlX, prev.y, point.x, point.y);
    }
    context.strokeStyle = "#1ea6f6";
    context.lineWidth = 3;
    context.stroke();

    context.fillStyle = "#1ea6f6";
    points.forEach((point) => {
        context.beginPath();
        context.arc(point.x, point.y, 4, 0, Math.PI * 2);
        context.fill();
    });

    context.fillStyle = "#7f8ca0";
    context.font = "12px Plus Jakarta Sans, sans-serif";
    dayKeys.forEach((key, index) => {
        const x = leftPadding + (chartWidth / (dayKeys.length - 1)) * index;
        const label = key.slice(5);
        context.fillText(label, x - 14, height - 14);
    });
});
