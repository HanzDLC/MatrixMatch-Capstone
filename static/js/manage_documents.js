document.addEventListener("DOMContentLoaded", () => {
    const searchInput = document.getElementById("documentSearch");
    const table = document.getElementById("documentsTable");
    const sortBtn = document.getElementById("sortDocumentsBtn");
    const filterBtn = document.getElementById("filterDocumentsBtn");

    // Pagination elements
    const prevBtn = document.getElementById("prevDocumentPage");
    const nextBtn = document.getElementById("nextDocumentPage");
    const pageInfo = document.getElementById("documentPageInfo");

    let showRecentOnly = false; // Note: For documents, maybe 'recent' implies last 30 IDs? Or just keep it symmetrical. We'll toggle it.
    let sortAsc = true;

    // Pagination state
    let currentPage = 1;
    const itemsPerPage = 10;

    if (!table) return;

    const tbody = table.querySelector("tbody");
    // Store all original rows to manipulate safely
    const allRows = Array.from(tbody.querySelectorAll("tr"));

    // Function to apply filters, sort, and pagination
    const updateTable = () => {
        const term = (searchInput?.value || "").toLowerCase().trim();

        // 1. Filter
        let filteredRows = allRows.filter((row) => {
            const text = row.innerText.toLowerCase();
            const passesSearch = text.includes(term);
            const passesFilter = !showRecentOnly; // Currently, filter doesn't do much for docs unless we filter by program
            return passesSearch && passesFilter;
        });

        // 2. Sort
        filteredRows.sort((a, b) => {
            const idA = parseInt(a.querySelector("td").innerText.trim(), 10);
            const idB = parseInt(b.querySelector("td").innerText.trim(), 10);
            return sortAsc ? (idA - idB) : (idB - idA);
        });

        // 3. Paginate
        const totalPages = Math.max(1, Math.ceil(filteredRows.length / itemsPerPage));
        if (currentPage > totalPages) {
            currentPage = totalPages;
        }

        const startIndex = (currentPage - 1) * itemsPerPage;
        const endIndex = startIndex + itemsPerPage;
        const paginatedRows = filteredRows.slice(startIndex, endIndex);

        // Render
        // Clear tbody safely
        while (tbody.firstChild) {
            tbody.removeChild(tbody.firstChild);
        }

        paginatedRows.forEach(row => {
            row.style.display = "";
            tbody.appendChild(row);
        });

        // Update Pagination Controls
        if (pageInfo) {
            pageInfo.textContent = `Page ${currentPage} of ${totalPages}`;
        }
        if (prevBtn) {
            prevBtn.disabled = currentPage <= 1;
        }
        if (nextBtn) {
            nextBtn.disabled = currentPage >= totalPages;
        }
    };

    // Event Listeners
    if (searchInput) {
        searchInput.addEventListener("input", () => {
            currentPage = 1; // Reset to first page on search
            updateTable();
        });
    }

    if (sortBtn) {
        sortBtn.addEventListener("click", () => {
            sortAsc = !sortAsc;
            sortBtn.textContent = sortAsc ? "Sort ID (Asc)" : "Sort ID (Desc)";
            updateTable();
        });
    }

    if (filterBtn) {
        filterBtn.addEventListener("click", () => {
            showRecentOnly = !showRecentOnly;
            filterBtn.textContent = showRecentOnly ? "Filters (Active)" : "Filters";
            currentPage = 1;
            updateTable();
        });
    }

    if (prevBtn) {
        prevBtn.addEventListener("click", () => {
            if (currentPage > 1) {
                currentPage--;
                updateTable();
            }
        });
    }

    if (nextBtn) {
        nextBtn.addEventListener("click", () => {
            currentPage++;
            updateTable();
        });
    }

    // Initial render
    updateTable();
});
