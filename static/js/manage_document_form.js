document.addEventListener("DOMContentLoaded", () => {
    const fileInput = document.getElementById("documentFileExtractor");
    const extractBtn = document.getElementById("extractBtn");
    const statusDiv = document.getElementById("extractStatus");

    // Form fields to auto-populate
    const titleField = document.getElementById("title");
    const abstractField = document.getElementById("abstract");

    if (!fileInput || !extractBtn || !statusDiv) return;

    extractBtn.addEventListener("click", async () => {
        const file = fileInput.files[0];

        if (!file) {
            statusDiv.textContent = "Please select a file first.";
            statusDiv.style.color = "var(--color-danger)";
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        // UI Feedback
        extractBtn.disabled = true;
        extractBtn.textContent = "Extracting...";
        statusDiv.textContent = "Processing document, please wait...";
        statusDiv.style.color = "var(--text-main)";

        try {
            const response = await fetch("/admin/documents/extract", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.error || "Failed to extract text from document.");
            }

            const data = await response.json();

            // Auto-populate fields
            if (data.title && titleField) {
                titleField.value = data.title;
            }
            if (data.abstract && abstractField) {
                abstractField.value = data.abstract;
            }

            statusDiv.textContent = "Extraction successful! Please review the auto-populated fields.";
            statusDiv.style.color = "var(--color-success)";

        } catch (error) {
            console.error("Extraction error:", error);
            statusDiv.textContent = error.message;
            statusDiv.style.color = "var(--color-danger)";
        } finally {
            extractBtn.disabled = false;
            extractBtn.textContent = "Extract Text";
        }
    });

    // Optional: Auto-trigger extraction when file is selected
    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            statusDiv.textContent = "File selected. Click 'Extract Text' to begin.";
            statusDiv.style.color = "var(--text-muted)";
        } else {
            statusDiv.textContent = "";
        }
    });
});
