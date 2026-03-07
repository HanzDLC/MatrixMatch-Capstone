const escapeHtml = (value = "") => String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");

const createListSignature = (items = [], projector) => items.map(projector).join("||");

const createMessageSignature = (messages = []) => createListSignature(
    messages,
    (message) => `${message.id}:${message.sender_id}:${message.timestamp || ""}:${message.content || ""}`
);

const createSerialPoller = (task, intervalMs) => {
    let timerId = null;
    let inFlightPromise = null;
    let shouldRerun = false;

    const run = () => {
        if (inFlightPromise) {
            shouldRerun = true;
            return inFlightPromise;
        }

        inFlightPromise = Promise.resolve()
            .then(task)
            .finally(() => {
                const rerunAfterCompletion = shouldRerun;
                shouldRerun = false;
                inFlightPromise = null;

                if (rerunAfterCompletion) {
                    run();
                }
            });

        return inFlightPromise;
    };

    return {
        run,
        start({ immediate = true } = {}) {
            if (timerId) {
                return;
            }

            if (immediate) {
                run();
            }

            timerId = window.setInterval(run, intervalMs);
        },
        stop() {
            if (timerId) {
                window.clearInterval(timerId);
                timerId = null;
            }

            shouldRerun = false;
        }
    };
};

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

    const setDropdownState = (dropdown, isOpen) => {
        if (!dropdown) return;
        dropdown.classList.toggle("is-open", isOpen);
        dropdown.setAttribute("aria-hidden", isOpen ? "false" : "true");
    };

    const closeDropdown = (dropdown) => {
        setDropdownState(dropdown, false);
    };

    const toggleExclusiveDropdown = (dropdownToToggle, ...dropdownsToClose) => {
        if (!dropdownToToggle) return false;

        const shouldOpen = !dropdownToToggle.classList.contains("is-open");
        dropdownsToClose.forEach(closeDropdown);
        setDropdownState(dropdownToToggle, shouldOpen);
        return shouldOpen;
    };

    const bindOutsideClickCloser = (button, dropdown) => {
        if (!button || !dropdown) return;

        document.addEventListener("click", (e) => {
            if (!button.contains(e.target) && !dropdown.contains(e.target)) {
                closeDropdown(dropdown);
            }
        });
    };

    const openChatHeadFromTrigger = (chatTrigger) => {
        if (!chatTrigger || typeof window.openChatHead !== "function") return;

        const userId = Number(chatTrigger.dataset.chatUserId);
        if (Number.isNaN(userId)) return;

        window.openChatHead(
            userId,
            chatTrigger.dataset.chatUserName || "",
            chatTrigger.dataset.chatProfilePic || "",
            chatTrigger.dataset.chatInitials || ""
        );
    };

    const createRecentMessagesSignature = (conversations, unreadCount) => `${unreadCount}::${createListSignature(
        conversations,
        (conversation) => [
            conversation.other_user_id,
            conversation.timestamp || "",
            conversation.is_read ? 1 : 0,
            conversation.was_sent_by_me ? 1 : 0,
            conversation.latest_message || ""
        ].join(":")
    )}`;

    const createUnreadConversationsSignature = (conversations) => createListSignature(
        conversations.filter((conversation) => !conversation.is_read && !conversation.was_sent_by_me),
        (conversation) => `${conversation.other_user_id}:${conversation.timestamp || ""}`
    );

    const createOnlineUsersSignature = (users) => createListSignature(
        users,
        (user) => [
            user.id,
            user.name || "",
            user.role || "",
            user.profile_pic || "",
            user.initials || "",
            user.is_online ? 1 : 0
        ].join(":")
    );

    // --- Document Alerts Logic ---
    const notifBtn = document.getElementById("notif-btn-toggle");
    const notifDot = notifBtn ? notifBtn.querySelector(".notif-btn__dot") : null;
    const notifDropdown = document.getElementById("notificationsDropdown");
    const markAllReadBtn = document.getElementById("markAllReadBtn");
    const notificationsList = document.getElementById("notificationsList");
    const msgBtn = document.getElementById("msg-btn-toggle");
    const msgDot = document.getElementById("msgDot");
    const msgDropdown = document.getElementById("messagesDropdown");
    const recentMessagesList = document.getElementById("recentMessagesList");

    if (notifBtn && notifDropdown) {
        // Hide dot initially
        if (notifDot) notifDot.style.display = "none";

        let lastSeenAlertId = parseInt(localStorage.getItem("matrixmatch_last_alert_id") || "0", 10);
        let latestAlertId = lastSeenAlertId;
        let hasNewAlerts = false;

        const getRelativeTime = (isoString) => {
            if (!isoString) return "";
            const date = new Date(isoString);
            const now = new Date();
            const diffInSeconds = Math.floor((now - date) / 1000);

            if (diffInSeconds < 60) return "just now";
            const diffInMinutes = Math.floor(diffInSeconds / 60);
            if (diffInMinutes < 60) return `${diffInMinutes}m ago`;
            const diffInHours = Math.floor(diffInMinutes / 60);
            if (diffInHours < 24) return `${diffInHours}h ago`;
            const diffInDays = Math.floor(diffInHours / 24);
            return `${diffInDays}d ago`;
        };

        const renderNotifications = (docs) => {
            if (!docs || docs.length === 0) {
                notificationsList.innerHTML = '<div class="notifications-empty">You\'re all caught up!</div>';
                return;
            }

            notificationsList.innerHTML = docs.map(doc => `
                <div class="notifications-item">
                    <div style="display: flex; align-items: start;">
                        <span class="notifications-item__icon">📄</span>
                        <div style="flex: 1;">
                            <div class="notifications-item__title">New Document Added</div>
                            <div class="notifications-item__title" style="font-weight: 400; color: var(--text-soft); font-size: 0.8rem; margin-top: 2px;">"${doc.title}"</div>
                            <div class="notifications-item__time">${getRelativeTime(doc.timestamp)}</div>
                        </div>
                    </div>
                </div>
            `).join('');
        };

        const markAlertsSeen = () => {
            if (notifDot) notifDot.style.display = "none";
            if (latestAlertId > lastSeenAlertId) {
                localStorage.setItem("matrixmatch_last_alert_id", latestAlertId.toString());
                lastSeenAlertId = latestAlertId;
            }
            hasNewAlerts = false;
        };

        notifBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            if (toggleExclusiveDropdown(notifDropdown, msgDropdown)) {
                markAlertsSeen();
            }
        });

        fetch("/api/alerts")
            .then(res => {
                if (!res.ok) throw new Error("API not okay");
                return res.json();
            })
            .then(data => {
                const recentDocuments = data.recent_documents || [];

                latestAlertId = Number(data.latest_log_id || lastSeenAlertId);
                hasNewAlerts = latestAlertId > lastSeenAlertId;

                if (hasNewAlerts) {
                    if (notifDropdown.classList.contains("is-open")) {
                        markAlertsSeen();
                    } else if (recentDocuments.length > 0 && notifDot) {
                        notifDot.style.display = "block";
                    }
                }

                renderNotifications(recentDocuments);
            })
            .catch(err => {
                console.error("Failed to fetch alerts", err);
                renderNotifications([]);
            });

        // Mark as read button simply closes it for now
        if (markAllReadBtn) {
            markAllReadBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                markAlertsSeen();
                closeDropdown(notifDropdown);
            });
        }

        bindOutsideClickCloser(notifBtn, notifDropdown);
    }

    // --- Messages Dropdown Logic ---
    if (msgBtn && msgDropdown && recentMessagesList) {
        let lastRecentMessagesSignature = null;
        let lastUnreadConversationsSignature = null;

        const renderMessages = (conversations) => {
            if (!conversations || conversations.length === 0) {
                recentMessagesList.innerHTML = '<div class="notifications-empty">No conversations yet</div>';
                return;
            }

            recentMessagesList.innerHTML = conversations.map(conv => {
                const fullName = `${conv.first_name} ${conv.last_name}`;
                const safeName = escapeHtml(fullName);
                const safeInitials = escapeHtml(conv.initials || "");
                const safeProfilePic = escapeHtml(conv.profile_pic || "");
                const previewText = escapeHtml(`${conv.was_sent_by_me ? 'You: ' : ''}${conv.latest_message || ''}`);
                const avatar = conv.profile_pic
                    ? `<img src="/static/img/uploads/profiles/${safeProfilePic}?v=1" alt="${safeName}" style="width:100%; height:100%; border-radius:50%; object-fit:cover;">`
                    : safeInitials;

                const isUnread = !conv.is_read && !conv.was_sent_by_me;
                const fontWeight = isUnread ? '700' : '400';
                const color = isUnread ? 'var(--text-main)' : 'var(--text-soft)';

                return `
                        <div class="notifications-item"
                            data-chat-user-id="${conv.other_user_id}"
                            data-chat-user-name="${safeName}"
                            data-chat-profile-pic="${safeProfilePic}"
                            data-chat-initials="${safeInitials}"
                            style="cursor: pointer;">
                            <div style="display: flex; align-items: center; gap: 12px;">
                                <div style="width:36px; height:36px; border-radius:50%; background:var(--accent-main); color:white; display:flex; align-items:center; justify-content:center; font-size:0.8rem; font-weight:700;">
                                    ${avatar}
                                </div>
                                <div style="flex: 1; overflow:hidden;">
                                    <div class="notifications-item__title" style="font-weight:600;">${safeName}</div>
                                    <div class="notifications-item__title" style="font-weight: ${fontWeight}; color: ${color}; font-size: 0.8rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                                        ${previewText}
                                    </div>
                                </div>
                                ${isUnread ? '<div style="width:8px; height:8px; border-radius:50%; background:var(--error-main);"></div>' : ''}
                            </div>
                        </div>
                `;
            }).join('');
        };

        const autoOpenUnreadChatHeads = (conversations) => {
            conversations.forEach(conv => {
                const isUnread = !conv.is_read && !conv.was_sent_by_me;
                if (isUnread && !document.getElementById(`chat-head-${conv.other_user_id}`) && typeof window.openChatHead === 'function') {
                    window.openChatHead(conv.other_user_id, `${conv.first_name} ${conv.last_name}`, conv.profile_pic, conv.initials);
                }
            });
        };

        const fetchRecentMessages = () => {
            return fetch("/api/messages/recent")
                .then(res => res.json())
                .then(data => {
                    const conversations = data.conversations || [];
                    const recentMessagesSignature = createRecentMessagesSignature(conversations, data.unread_count || 0);
                    const unreadConversationsSignature = createUnreadConversationsSignature(conversations);

                    if (msgDot) {
                        msgDot.style.display = data.unread_count > 0 ? "flex" : "none";
                        msgDot.textContent = data.unread_count > 0 ? data.unread_count : "";
                    }

                    if (recentMessagesSignature !== lastRecentMessagesSignature) {
                        renderMessages(conversations);
                        lastRecentMessagesSignature = recentMessagesSignature;
                    }

                    if (unreadConversationsSignature !== lastUnreadConversationsSignature) {
                        autoOpenUnreadChatHeads(conversations);
                        lastUnreadConversationsSignature = unreadConversationsSignature;
                    }
                })
                .catch(err => console.error("Failed to fetch recent messages", err));
        };

        const recentMessagesPoller = createSerialPoller(fetchRecentMessages, 15000);

        // Fetch immediately and poll every 15s
        recentMessagesPoller.start();

        // Toggle dropdown
        msgBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            if (toggleExclusiveDropdown(msgDropdown, notifDropdown)) {
                recentMessagesPoller.run(); // Refresh on open
            }
        });

        recentMessagesList.addEventListener("click", (e) => {
            const chatTrigger = e.target.closest(".notifications-item[data-chat-user-id]");
            if (chatTrigger && recentMessagesList.contains(chatTrigger)) {
                openChatHeadFromTrigger(chatTrigger);
            }
        });

        bindOutsideClickCloser(msgBtn, msgDropdown);
    }

    // --- Custom Confirmation Modal Logic ---
    // --- Profile Picture Upload Logic ---
    const avatarTrigger = document.getElementById("avatarUploadTrigger");
    const fileInput = document.getElementById("profilePicInput");

    if (avatarTrigger && fileInput) {
        avatarTrigger.addEventListener("click", () => fileInput.click());

        fileInput.addEventListener("change", function () {
            if (this.files && this.files[0]) {
                const formData = new FormData();
                formData.append("file", this.files[0]);

                // Show a simple loading state or SweetAlert
                Swal.fire({
                    title: 'Uploading...',
                    allowOutsideClick: false,
                    didOpen: () => {
                        Swal.showLoading();
                    }
                });

                fetch("/profile/upload_pic", {
                    method: "POST",
                    body: formData
                })
                    .then(res => res.json())
                    .then(data => {
                        if (data.success) {
                            Swal.fire({
                                icon: 'success',
                                title: 'Success',
                                text: 'Profile picture updated!',
                                timer: 2000,
                                showConfirmButton: false
                            }).then(() => {
                                window.location.reload(); // Reload to update all avatars
                            });
                        } else {
                            Swal.fire('Error', data.error || 'Upload failed', 'error');
                        }
                    })
                    .catch(err => {
                        console.error(err);
                        Swal.fire('Error', 'An error occurred during upload', 'error');
                    });
            }
        });
    }
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

    // --- Sidebar Users Polling & Search ---
    const onlineUsersList = document.getElementById("onlineUsersList");
    const sidebarUserSearch = document.getElementById("sidebarUserSearch");

    if (onlineUsersList) {
        let allSidebarUsers = []; // Store fetched users for local filtering
        let lastOnlineUsersSignature = null;

        const renderSidebarUsers = (filterText = "") => {
            if (allSidebarUsers.length === 0) {
                onlineUsersList.innerHTML = '<li style="font-size: 0.8rem; color: var(--text-soft); padding: 0 8px;">No users found.</li>';
                return;
            }

            const lowerFilter = filterText.toLowerCase();
            const filteredUsers = allSidebarUsers.filter(user => user.name.toLowerCase().includes(lowerFilter));

            if (filteredUsers.length === 0) {
                onlineUsersList.innerHTML = '<li style="font-size: 0.8rem; color: var(--text-soft); padding: 0 8px;">No matches found.</li>';
                return;
            }

            onlineUsersList.innerHTML = filteredUsers.map(user => {
                const safeName = escapeHtml(user.name);
                const safeProfilePic = escapeHtml(user.profile_pic || "");
                const safeInitials = escapeHtml(user.initials || "");
                const avatarContent = user.profile_pic
                    ? `<img src="/static/img/uploads/profiles/${safeProfilePic}?v=1" alt="${safeName}">`
                    : safeInitials;

                const badge = user.role === 'Admin'
                    ? `<span class="online-user-role-badge">Admin</span>`
                    : '';

                // Green for online, gray for offline
                const dotColor = user.is_online ? "#12b76a" : "#747f8d";
                const dotShadow = user.is_online ? "0 0 3px rgba(18,183,106,0.5)" : "none";

                return `
                    <li class="online-user-item"
                        data-chat-user-id="${user.id}"
                        data-chat-user-name="${safeName}"
                        data-chat-profile-pic="${safeProfilePic}"
                        data-chat-initials="${safeInitials}"
                        style="cursor: pointer; opacity: ${user.is_online ? '1' : '0.65'};">
                        <div class="online-user-avatar" style="${!user.is_online ? 'filter: grayscale(1);' : ''}">
                            ${avatarContent}
                            <div class="online-user-dot" style="background: ${dotColor}; box-shadow: ${dotShadow}; border-color: ${user.is_online ? 'var(--bg-card)' : 'var(--bg-main)'};"></div>
                        </div>
                        <div class="online-user-name">
                            ${safeName}
                            ${badge}
                        </div>
                    </li>
                `;
            }).join('');
        };

        const fetchOnlineUsers = () => {
            return fetch("/api/online_users")
                .then(res => res.json())
                .then(data => {
                    const users = data.online_users || [];
                    const usersSignature = createOnlineUsersSignature(users);

                    if (usersSignature === lastOnlineUsersSignature) {
                        return;
                    }

                    lastOnlineUsersSignature = usersSignature;
                    allSidebarUsers = users;
                    // Render using the current search filter
                    const currentFilter = sidebarUserSearch ? sidebarUserSearch.value : "";
                    renderSidebarUsers(currentFilter);
                })
                .catch(err => console.error("Could not fetch sidebar users:", err));
        };

        const onlineUsersPoller = createSerialPoller(fetchOnlineUsers, 60000);

        if (sidebarUserSearch) {
            sidebarUserSearch.addEventListener('input', (e) => {
                renderSidebarUsers(e.target.value);
            });
        }

        onlineUsersList.addEventListener("click", (e) => {
            const chatTrigger = e.target.closest(".online-user-item[data-chat-user-id]");
            if (chatTrigger && onlineUsersList.contains(chatTrigger)) {
                openChatHeadFromTrigger(chatTrigger);
            }
        });

        // Fetch immediately, then poll every 60 seconds (slower polling for full list)
        onlineUsersPoller.start();
    }
});

// --- Floating Chat Heads Logic ---
window.activeChatIntervals = window.activeChatIntervals || {};
window.chatHeadUsageOrder = window.chatHeadUsageOrder || [];
const MAX_OPEN_CHAT_HEADS = 3;

const getChatHeadUserId = (chatBox) => {
    if (!chatBox) return null;

    const rawUserId = chatBox.dataset.userId || chatBox.id.replace("chat-head-", "");
    const parsedUserId = Number(rawUserId);
    return Number.isNaN(parsedUserId) ? null : parsedUserId;
};

const removeChatHeadFromUsage = (userId) => {
    const usageIndex = window.chatHeadUsageOrder.indexOf(userId);
    if (usageIndex !== -1) {
        window.chatHeadUsageOrder.splice(usageIndex, 1);
    }
};

const touchChatHead = (userId) => {
    removeChatHeadFromUsage(userId);
    window.chatHeadUsageOrder.push(userId);
};

const stopChatHeadPolling = (userId) => {
    const activePoller = window.activeChatIntervals[userId];
    if (!activePoller) {
        return;
    }

    if (typeof activePoller.stop === "function") {
        activePoller.stop();
    } else {
        clearInterval(activePoller);
    }

    delete window.activeChatIntervals[userId];
};

const closeChatHead = (userId) => {
    const chatBox = document.getElementById(`chat-head-${userId}`);

    stopChatHeadPolling(userId);

    removeChatHeadFromUsage(userId);

    if (chatBox) {
        chatBox.remove();
    }
};

const enforceChatHeadLimit = (container) => {
    while (container.children.length >= MAX_OPEN_CHAT_HEADS) {
        const oldestChatUserId = window.chatHeadUsageOrder[0] ?? getChatHeadUserId(container.querySelector(".chat-head-box"));
        if (oldestChatUserId === null) {
            break;
        }
        closeChatHead(oldestChatUserId);
    }
};

window.openChatHead = function (userId, userName, profilePic, initials) {
    const container = document.getElementById("chat-heads-container");
    if (!container) return;

    const chatUserId = Number(userId);

    // Check if already open
    const existingChat = document.getElementById(`chat-head-${chatUserId}`);
    if (existingChat) {
        touchChatHead(chatUserId);

        // Just focus the input
        const inputField = existingChat.querySelector(`#chat-input-${chatUserId}`);
        if (inputField) inputField.focus();

        // Visual cue (brief shake or highlight)
        existingChat.style.transform = "scale(1.02)";
        setTimeout(() => existingChat.style.transform = "scale(1)", 200);
        return;
    }

    enforceChatHeadLimit(container);

    const safeUserName = escapeHtml(userName || "");
    const safeProfilePic = escapeHtml(profilePic || "");
    const safeInitials = escapeHtml(initials || "");
    const avatarHtml = profilePic
        ? `<img src="/static/img/uploads/profiles/${safeProfilePic}?v=1" alt="${safeUserName}">`
        : safeInitials;

    // Create the DOM element
    const chatBox = document.createElement("div");
    chatBox.className = "chat-head-box";
    chatBox.id = `chat-head-${chatUserId}`;
    chatBox.dataset.userId = String(chatUserId);
    chatBox.innerHTML = `
        <div class="chat-head-header">
            <div class="chat-head-user">
                <div class="chat-head-avatar">${avatarHtml}</div>
                <span>${safeUserName}</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <button class="chat-head-minimize" aria-label="Minimize chat" title="Minimize" style="background:none; border:none; color:white; cursor:pointer; font-size: 1.2rem; line-height: 1; opacity: 0.8; padding: 0;">-</button>
                <button class="chat-head-close" aria-label="Close chat" title="Close" style="background:none; border:none; color:white; cursor:pointer; font-size: 1.2rem; line-height: 1; opacity: 0.8; padding: 0;">&times;</button>
            </div>
        </div>
        <div class="chat-head-content" id="chat-content-${chatUserId}" style="display: flex; flex-direction: column; flex: 1; overflow: hidden;">
            <div class="chat-head-messages" id="chat-messages-${chatUserId}">
                <div class="chat-head-placeholder" style="text-align:center; padding: 20px; color: var(--text-soft); font-size: 0.8rem;">Loading...</div>
            </div>
            <div class="chat-head-input" style="padding: 8px 10px; display: flex; align-items: center; gap: 6px;">
                <input type="text" id="chat-input-${chatUserId}" placeholder="Type..." aria-label="Type a message" style="flex: 1; min-width: 0;">
                <button class="chat-head-send-btn" title="Send message" style="background: var(--accent-main); color: white; border: none; border-radius: 4px; padding: 6px 10px; cursor: pointer; font-size: 0.8rem; font-weight: 700;">Send</button>
            </div>
        </div>
    `;

    container.appendChild(chatBox);
    touchChatHead(chatUserId);

    const messagesContainer = chatBox.querySelector(`#chat-messages-${chatUserId}`);
    const contentArea = chatBox.querySelector(`#chat-content-${chatUserId}`);
    const closeBtn = chatBox.querySelector(".chat-head-close");
    const minimizeBtn = chatBox.querySelector(".chat-head-minimize");
    const sendBtn = chatBox.querySelector(".chat-head-send-btn");
    const inputField = chatBox.querySelector(`#chat-input-${chatUserId}`);
    const header = chatBox.querySelector(".chat-head-header");
    let lastRenderedMessageSignature = "";
    let hasPendingOptimisticMessage = false;
    let failedMessageResetTimeoutId = null;
    const clearFailedMessageReset = () => {
        if (failedMessageResetTimeoutId) {
            window.clearTimeout(failedMessageResetTimeoutId);
            failedMessageResetTimeoutId = null;
        }
    };
    const scheduleFailedMessageReset = () => {
        clearFailedMessageReset();
        failedMessageResetTimeoutId = window.setTimeout(() => {
            failedMessageResetTimeoutId = null;
            if (chatBox.isConnected) {
                lastRenderedMessageSignature = "";
                chatMessagesPoller.run();
            }
        }, 2000);
    };
    const renderChatPlaceholder = (message) => {
        messagesContainer.innerHTML = `<div class="chat-head-placeholder" style="text-align:center; padding: 20px; color: var(--text-soft); font-size: 0.8rem;">${escapeHtml(message)}</div>`;
    };
    const renderChatMessages = (messages) => {
        messagesContainer.innerHTML = messages.map(msg => {
            const isSent = msg.sender_id !== chatUserId;
            const bubbleClass = isSent ? 'sent' : 'received';
            return `
                <div class="chat-bubble ${bubbleClass}">${escapeHtml(msg.content || "")}</div>
                <div class="chat-timestamp">${formatTime(msg.timestamp)}</div>
            `;
        }).join('');
    };

    // Toggle Minimize
    const toggleMinimize = () => {
        const isMinimized = contentArea.style.display === "none";
        contentArea.style.display = isMinimized ? "flex" : "none";
        chatBox.style.height = isMinimized ? "400px" : "auto";
        chatBox.style.width = isMinimized ? "320px" : "200px";
        if (isMinimized) {
            setTimeout(() => inputField.focus(), 50);
        }
    };

    minimizeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        touchChatHead(chatUserId);
        toggleMinimize();
    });

    header.addEventListener("click", () => {
        touchChatHead(chatUserId);
        if (contentArea.style.display === "none") {
            toggleMinimize();
        }
    });

    // Helper to format timestamps
    const formatTime = (isoString) => {
        const d = new Date(isoString);
        return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    };

    // Load conversation history
    const loadMessages = () => {
        return fetch(`/api/messages/conversation/${chatUserId}`)
            .then(res => res.json())
            .then(data => {
                if (data.error) throw new Error(data.error);

                const messages = data.messages || [];
                const messageSignature = createMessageSignature(messages);
                const shouldStickToBottom = !lastRenderedMessageSignature
                    || messagesContainer.scrollHeight - messagesContainer.clientHeight <= messagesContainer.scrollTop + 20;

                if (hasPendingOptimisticMessage && messageSignature === lastRenderedMessageSignature) {
                    return;
                }

                if (messageSignature === lastRenderedMessageSignature) {
                    return;
                }

                if (messages.length === 0) {
                    renderChatPlaceholder("Start of conversation");
                } else {
                    renderChatMessages(messages);
                }

                lastRenderedMessageSignature = messageSignature;
                hasPendingOptimisticMessage = false;

                if (shouldStickToBottom) {
                    messagesContainer.scrollTop = messagesContainer.scrollHeight;
                }
            })
            .catch(err => {
                console.error("Error loading chat messages:", err);
            });
    };

    const chatMessagesPoller = createSerialPoller(loadMessages, 5000);
    window.activeChatIntervals[chatUserId] = chatMessagesPoller;
    chatMessagesPoller.start();

    // Close Handler
    closeBtn.addEventListener("click", () => {
        clearFailedMessageReset();
        closeChatHead(chatUserId);
    });

    // Send Message Handler
    const sendMessage = () => {
        if (inputField.value.trim() === "") return;

        const tempVal = inputField.value.trim();
        inputField.value = "";

        // Optimistic UI update
        const bubble = document.createElement("div");
        bubble.className = "chat-bubble sent";
        bubble.textContent = tempVal;

        const time = document.createElement("div");
        time.className = "chat-timestamp";
        time.textContent = "Sending...";

        if (messagesContainer.querySelector(".chat-head-placeholder")) {
            messagesContainer.innerHTML = '';
        }
        messagesContainer.appendChild(bubble);
        messagesContainer.appendChild(time);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        touchChatHead(chatUserId);
        hasPendingOptimisticMessage = true;
        clearFailedMessageReset();

        // Transmit API call
        fetch("/api/messages/send", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ receiver_id: chatUserId, content: tempVal })
        })
            .then(res => res.json())
            .then(data => {
                if (data.error) {
                    hasPendingOptimisticMessage = false;
                    time.textContent = "Failed to send";
                    time.style.color = "red";
                    scheduleFailedMessageReset();
                } else {
                    chatMessagesPoller.run(); // reload ground truth
                }
            })
            .catch(err => {
                console.error("Error sending message:", err);
                hasPendingOptimisticMessage = false;
                time.textContent = "Failed to send";
                time.style.color = "red";
                scheduleFailedMessageReset();
            });
    };

    inputField.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
            sendMessage();
        }
    });

    if (sendBtn) {
        sendBtn.addEventListener("click", sendMessage);
    }

    // Focus input automatically
    setTimeout(() => inputField.focus(), 100);
};
