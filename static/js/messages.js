document.addEventListener("DOMContentLoaded", () => {
    const conversationList = document.getElementById("conversationList");
    const chatHeader = document.getElementById("chatHeader");
    const chatBody = document.getElementById("chatBody");
    const chatInputArea = document.getElementById("chatInputArea");
    const activeChatName = document.getElementById("activeChatName");
    const activeChatAvatar = document.getElementById("activeChatAvatar");
    const messageInput = document.getElementById("messageInput");
    const sendMessageBtn = document.getElementById("sendMessageBtn");

    let activeUserId = null;
    let activeThreadItem = null;
    let pollInterval = null;
    let renderedConversationSignature = "";
    const inFlightConversationRequests = new Map();
    const queuedConversationRefreshOptions = new Map();

    const escapeHtml = (value = "") => String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");

    const createMessageSignature = (messages = []) => messages
        .map(message => `${message.id}:${message.sender_id}:${message.timestamp || ""}:${message.content || ""}`)
        .join("||");

    const scrollToBottom = () => {
        if (chatBody) {
            chatBody.scrollTop = chatBody.scrollHeight;
        }
    };

    const isChatBodyNearBottom = () => {
        if (!chatBody) {
            return true;
        }

        return chatBody.scrollHeight - chatBody.clientHeight <= chatBody.scrollTop + 24;
    };

    const setComposerState = (isBusy) => {
        if (!messageInput || !sendMessageBtn) {
            return;
        }

        const shouldDisable = isBusy || !activeUserId;
        messageInput.disabled = shouldDisable;
        sendMessageBtn.disabled = shouldDisable;
    };

    const renderConversation = (messages, { forceScroll = false } = {}) => {
        if (!chatBody) {
            return;
        }

        const shouldStickToBottom = forceScroll || !renderedConversationSignature || isChatBodyNearBottom();

        if (messages.length === 0) {
            chatBody.innerHTML = '<div style="margin:auto; color:var(--text-soft);">No messages yet. Say hello!</div>';
            return;
        }

        chatBody.innerHTML = messages.map(msg => {
            const isSentByMe = msg.sender_id === currentUserId;
            const bubbleClass = isSentByMe ? 'sent' : 'received';
            const timeString = msg.timestamp ? new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';

            return `
                <div class="chat-bubble ${bubbleClass}">
                    ${escapeHtml(msg.content || "")}
                </div>
                <div class="chat-timestamp">${timeString}</div>
            `;
        }).join('');

        if (shouldStickToBottom) {
            scrollToBottom();
        }
    };

    const loadConversation = (userId, { forceRender = false, forceScroll = false } = {}) => {
        if (!userId) {
            return Promise.resolve();
        }

        const requestKey = String(userId);
        const existingRequest = inFlightConversationRequests.get(requestKey);
        if (existingRequest) {
            if (forceRender || forceScroll) {
                const existingOptions = queuedConversationRefreshOptions.get(requestKey) || { forceRender: false, forceScroll: false };
                queuedConversationRefreshOptions.set(requestKey, {
                    forceRender: existingOptions.forceRender || forceRender,
                    forceScroll: existingOptions.forceScroll || forceScroll
                });
            }
            return existingRequest;
        }

        const request = fetch(`/api/messages/conversation/${userId}`)
            .then(res => res.json())
            .then(data => {
                if (String(activeUserId) !== requestKey) {
                    return;
                }

                const messages = data.messages || [];
                const messageSignature = createMessageSignature(messages);

                if (!forceRender && messageSignature === renderedConversationSignature) {
                    return;
                }

                renderedConversationSignature = messageSignature;
                renderConversation(messages, { forceScroll });
            })
            .catch(err => console.error("Error loading conversation", err))
            .finally(() => {
                if (inFlightConversationRequests.get(requestKey) === request) {
                    inFlightConversationRequests.delete(requestKey);
                }

                const queuedRefresh = queuedConversationRefreshOptions.get(requestKey);
                if (queuedRefresh) {
                    queuedConversationRefreshOptions.delete(requestKey);
                    if (String(activeUserId) === requestKey) {
                        loadConversation(userId, queuedRefresh);
                    }
                }
            });

        inFlightConversationRequests.set(requestKey, request);
        return request;
    };

    const setupChatView = (item) => {
        if (activeThreadItem && activeThreadItem !== item) {
            activeThreadItem.classList.remove("is-active");
        }

        activeThreadItem = item;
        item.classList.add("is-active");

        activeUserId = item.getAttribute("data-other-id");
        renderedConversationSignature = "";
        activeChatName.textContent = item.getAttribute("data-other-name");
        activeChatAvatar.innerHTML = item.querySelector(".online-user-avatar").innerHTML;

        chatHeader.style.display = "flex";
        chatInputArea.style.display = "flex";
        setComposerState(false);

        chatBody.innerHTML = '<div style="margin:auto; color:var(--text-soft);">Loading messages...</div>';

        loadConversation(activeUserId, { forceRender: true, forceScroll: true });

        if (pollInterval) {
            clearInterval(pollInterval);
        }

        pollInterval = setInterval(() => {
            if (activeUserId) {
                loadConversation(activeUserId);
            }
        }, 5000);
    };

    if (conversationList) {
        conversationList.addEventListener("click", (e) => {
            const threadItem = e.target.closest(".msg-thread-item");
            if (threadItem && conversationList.contains(threadItem)) {
                setupChatView(threadItem);
            }
        });
    }

    const sendMessage = () => {
        const content = messageInput?.value?.trim();
        if (!content || !activeUserId) {
            return;
        }

        const targetUserId = activeUserId;
        setComposerState(true);

        fetch("/api/messages/send", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ receiver_id: targetUserId, content: content })
        })
            .then(res => res.json())
            .then(data => {
                if (String(activeUserId) !== String(targetUserId)) {
                    return;
                }

                setComposerState(false);
                if (messageInput) {
                    messageInput.value = "";
                    messageInput.focus();
                }

                if (data.success) {
                    renderedConversationSignature = "";
                    loadConversation(targetUserId, { forceRender: true, forceScroll: true });
                }
            })
            .catch(err => {
                console.error(err);
            })
            .finally(() => {
                if (String(activeUserId) === String(targetUserId)) {
                    setComposerState(false);
                }
            });
    };

    sendMessageBtn?.addEventListener("click", sendMessage);
    messageInput?.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            sendMessage();
        }
    });
});
