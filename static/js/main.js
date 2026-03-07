document.addEventListener("DOMContentLoaded", () => {
    const sidebar = document.getElementById("sidebar");
    const sidebarToggle = document.getElementById("sidebarToggle");
    const sidebarBackdrop = document.getElementById("sidebarBackdrop");
    const themeToggle = document.getElementById("themeToggle");
    const themeToggleIcon = document.getElementById("themeToggleIcon");
    const backToTop = document.getElementById("backToTop");
    const body = document.body;
    const isMobileViewport = () => window.matchMedia("(max-width: 920px)").matches;

    const applyResponsiveTableLabels = () => {
        document.querySelectorAll(".table").forEach((table) => {
            const headers = Array.from(table.querySelectorAll("thead th")).map((header) =>
                header.textContent.trim().replace(/\s+/g, " ")
            );

            table.querySelectorAll("tbody tr").forEach((row) => {
                Array.from(row.children).forEach((cell, index) => {
                    if (cell.tagName !== "TD") {
                        return;
                    }

                    const label = headers[index] || row.dataset.label || "";
                    if (label) {
                        cell.dataset.label = label;
                    }
                });
            });
        });
    };

    const initializeDataTables = () => {
        if (!window.jQuery || !window.jQuery.fn || !window.jQuery.fn.DataTable) {
            return;
        }

        document.querySelectorAll('table[data-datatable="true"]').forEach((table) => {
            if (window.jQuery.fn.DataTable.isDataTable(table)) {
                return;
            }

            const options = {};
            const orderAttr = (table.dataset.datatableOrder || "").trim();
            if (orderAttr) {
                const [columnRaw, directionRaw] = orderAttr.split(",");
                const columnIndex = Number.parseInt(columnRaw, 10);
                if (Number.isInteger(columnIndex)) {
                    const direction = (directionRaw || "asc").trim().toLowerCase() === "desc" ? "desc" : "asc";
                    options.order = [[columnIndex, direction]];
                }
            }

            const emptyMessage = (table.dataset.datatableEmpty || "").trim();
            if (emptyMessage) {
                options.language = { emptyTable: emptyMessage };
            }

            window.jQuery(table).DataTable(options);
        });
    };

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
        const syncSidebarState = (isOpen) => {
            const open = Boolean(isOpen);
            sidebar.classList.toggle("sidebar--open", open);
            body.classList.toggle("has-sidebar-open", open && isMobileViewport());
            sidebarToggle.setAttribute("aria-expanded", open ? "true" : "false");
            sidebar.setAttribute("aria-hidden", open || !isMobileViewport() ? "false" : "true");

            if (sidebarBackdrop) {
                sidebarBackdrop.hidden = !open || !isMobileViewport();
                sidebarBackdrop.classList.toggle("is-visible", open && isMobileViewport());
            }
        };

        const closeSidebar = () => syncSidebarState(false);

        sidebarToggle.addEventListener("click", () => {
            syncSidebarState(!sidebar.classList.contains("sidebar--open"));
        });

        sidebarBackdrop?.addEventListener("click", closeSidebar);

        sidebar.querySelectorAll("a").forEach((link) => {
            link.addEventListener("click", () => {
                if (isMobileViewport()) {
                    closeSidebar();
                }
            });
        });

        document.addEventListener("click", (event) => {
            if (!isMobileViewport() || !sidebar.classList.contains("sidebar--open")) {
                return;
            }
            if (sidebar.contains(event.target) || sidebarToggle.contains(event.target)) {
                return;
            }
            closeSidebar();
        });

        document.addEventListener("keydown", (event) => {
            if (event.key === "Escape" && sidebar.classList.contains("sidebar--open")) {
                closeSidebar();
            }
        });

        window.addEventListener("resize", () => {
            if (!isMobileViewport()) {
                closeSidebar();
            } else {
                syncSidebarState(sidebar.classList.contains("sidebar--open"));
            }
        });

        syncSidebarState(false);
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

    if (window.jQuery) {
        initializeDataTables();
        window.jQuery(document).on("draw.dt", applyResponsiveTableLabels);
    }
    applyResponsiveTableLabels();

    // --- Document Alerts Logic ---
    const notifBtn = document.querySelector(".notif-btn");
    const notifDot = document.querySelector(".notif-btn__dot");
    const notifDropdown = document.getElementById("notificationsDropdown");
    const markAllReadBtn = document.getElementById("markAllReadBtn");
    const notificationsList = document.getElementById("notificationsList");

    if (notifBtn && notifDropdown) {
        // Hide dot initially
        if (notifDot) notifDot.style.display = "none";

        let latestAlertId = parseInt(localStorage.getItem("matrixmatch_last_alert_id") || "0", 10);
        let hasNewAlerts = false;

        const renderNotifications = (docs) => {
            if (!docs || docs.length === 0) {
                notificationsList.innerHTML = '<div class="notifications-empty">You\'re all caught up!</div>';
                return;
            }

            notificationsList.innerHTML = docs.map(doc => `
                <div class="notifications-item">
                    <div style="display: flex; align-items: start;">
                        <span class="notifications-item__icon">📄</span>
                        <div>
                            <div class="notifications-item__title">New Document Added</div>
                            <div class="notifications-item__title" style="font-weight: 400; color: var(--text-soft);">"${doc}"</div>
                        </div>
                    </div>
                </div>
            `).join('');
        };

        fetch("/api/alerts")
            .then(res => {
                if (!res.ok) throw new Error("API not okay");
                return res.json();
            })
            .then(data => {
                hasNewAlerts = data.latest_log_id > latestAlertId;

                if (hasNewAlerts && data.recent_documents.length > 0 && notifDot) {
                    notifDot.style.display = "block";
                }

                renderNotifications(data.recent_documents || []);

                // Toggle dropdown manually
                notifBtn.addEventListener("click", (e) => {
                    e.stopPropagation();
                    const isOpen = notifDropdown.classList.contains("is-open");

                    if (!isOpen) {
                        notifDropdown.classList.add("is-open");
                        notifDropdown.setAttribute("aria-hidden", "false");

                        // Clear the dot when they open it
                        if (notifDot) notifDot.style.display = "none";
                        if (hasNewAlerts) {
                            localStorage.setItem("matrixmatch_last_alert_id", data.latest_log_id.toString());
                            hasNewAlerts = false;
                        }
                    } else {
                        notifDropdown.classList.remove("is-open");
                        notifDropdown.setAttribute("aria-hidden", "true");
                    }
                });
            })
            .catch(err => {
                console.error("Failed to fetch alerts", err);
                renderNotifications([]);
                notifBtn.addEventListener("click", (e) => {
                    e.stopPropagation();
                    notifDropdown.classList.toggle("is-open");
                });
            });

        // Mark as read button simply closes it for now
        if (markAllReadBtn) {
            markAllReadBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                if (notifDot) notifDot.style.display = "none";
                notifDropdown.classList.remove("is-open");
            });
        }

        // Close when clicking outside
        document.addEventListener("click", (e) => {
            if (!notifBtn.contains(e.target) && !notifDropdown.contains(e.target)) {
                notifDropdown.classList.remove("is-open");
                notifDropdown.setAttribute("aria-hidden", "true");
            }
        });
    }

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
