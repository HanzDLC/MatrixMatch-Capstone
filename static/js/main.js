console.log("MatrixMatch frontend loaded.");

document.addEventListener("DOMContentLoaded", () => {
    const sidebar = document.getElementById("sidebar");
    const sidebarToggle = document.getElementById("sidebarToggle");
    const themeToggle = document.getElementById("themeToggle");
    const themeToggleIcon = document.getElementById("themeToggleIcon");
    const backToTop = document.getElementById("backToTop");
    const body = document.body;

    const preferredTheme = window.matchMedia &&
        window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light";

    const applyTheme = (theme) => {
        const safeTheme = theme === "dark" ? "dark" : "light";
        body.setAttribute("data-theme", safeTheme);
        if (themeToggleIcon) {
            themeToggleIcon.textContent = safeTheme === "dark" ? "Dark" : "Light";
        }
    };

    const savedTheme = localStorage.getItem("matrixmatch-theme");
    applyTheme(savedTheme || preferredTheme);

    if (themeToggle) {
        themeToggle.addEventListener("click", () => {
            const current = body.getAttribute("data-theme") === "dark" ? "dark" : "light";
            const next = current === "dark" ? "light" : "dark";
            localStorage.setItem("matrixmatch-theme", next);
            applyTheme(next);
        });
    }

    if (sidebar && sidebarToggle) {
        sidebarToggle.addEventListener("click", () => {
            sidebar.classList.toggle("sidebar--open");
        });

        document.addEventListener("click", (event) => {
            if (window.innerWidth > 920) {
                return;
            }
            if (!sidebar.classList.contains("sidebar--open")) {
                return;
            }
            if (sidebar.contains(event.target) || sidebarToggle.contains(event.target)) {
                return;
            }
            sidebar.classList.remove("sidebar--open");
        });
    }

    if (backToTop) {
        const updateBackToTop = () => {
            if (window.scrollY > 340) {
                backToTop.classList.add("is-visible");
            } else {
                backToTop.classList.remove("is-visible");
            }
        };
        window.addEventListener("scroll", updateBackToTop, { passive: true });
        updateBackToTop();
        backToTop.addEventListener("click", () => {
            window.scrollTo({ top: 0, behavior: "smooth" });
        });
    }

    document.querySelectorAll(".flash").forEach((flash) => {
        window.setTimeout(() => {
            flash.classList.add("is-dismissing");
            window.setTimeout(() => {
                flash.remove();
            }, 220);
        }, 5000);
    });

    document.querySelectorAll("form").forEach((form) => {
        form.addEventListener("submit", (event) => {
            const submitButton = form.querySelector('button[type="submit"]');
            if (!submitButton || submitButton.dataset.noLoading === "true") {
                return;
            }
            window.setTimeout(() => {
                if (event.defaultPrevented) {
                    return;
                }
                submitButton.dataset.originalText = submitButton.textContent || "";
                submitButton.textContent = "Processing...";
                submitButton.classList.add("is-loading");
                submitButton.disabled = true;
            }, 0);
        });
    });

    // --- Custom Confirmation Modal Logic ---
    const confirmModal = document.getElementById("confirmModal");
    const confirmModalTitle = document.getElementById("confirmModalTitle");
    const confirmModalMessage = document.getElementById("confirmModalMessage");
    const confirmModalAccept = document.getElementById("confirmModalAccept");
    let activeConfirmForm = null;

    const closeConfirmModal = () => {
        if (confirmModal) {
            confirmModal.classList.remove("is-open");
            confirmModal.setAttribute("aria-hidden", "true");
        }
        activeConfirmForm = null;
    };

    if (confirmModal) {
        document.querySelectorAll("[data-modal-close]").forEach(btn => {
            btn.addEventListener("click", closeConfirmModal);
        });

        confirmModalAccept?.addEventListener("click", () => {
            if (activeConfirmForm) {
                // Remove data-confirm so it doesn't trigger again, then submit
                activeConfirmForm.removeAttribute("data-confirm");
                activeConfirmForm.submit();
            }
            closeConfirmModal();
        });
    }

    document.querySelectorAll("form[data-confirm]").forEach((form) => {
        form.addEventListener("submit", (event) => {
            // If the attribute is still there, intercept it
            if (form.hasAttribute("data-confirm")) {
                event.preventDefault();
                event.stopImmediatePropagation();
                activeConfirmForm = form;

                const customMessage = form.getAttribute("data-confirm") || "Are you sure you want to proceed?";
                if (confirmModalMessage) {
                    confirmModalMessage.textContent = customMessage;
                }

                if (confirmModal) {
                    confirmModal.classList.add("is-open");
                    confirmModal.setAttribute("aria-hidden", "false");
                    // Focus the cancel button for safety
                    const cancelBtn = confirmModal.querySelector('.btn-secondary');
                    if (cancelBtn) cancelBtn.focus();
                }
            }
        });
    });

});
