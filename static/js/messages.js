document.addEventListener("DOMContentLoaded", () => {
    const threadItems = document.querySelectorAll(".msg-thread-item");
    const chatHeader = document.getElementById("chatHeader");
    const chatBody = document.getElementById("chatBody");
    const chatInputArea = document.getElementById("chatInputArea");
    const activeChatName = document.getElementById("activeChatName");
    const activeChatAvatar = document.getElementById("activeChatAvatar");
    const messageInput = document.getElementById("messageInput");
    const sendMessageBtn = document.getElementById("sendMessageBtn");

    let activeUserId = null;
    let pollInterval = null;

    const scrollToBottom = () => {
        chatBody.scrollTop = chatBody.scrollHeight;
    };

    const loadConversation = (userId) => {
        fetch(`/api/messages/conversation/${userId}`)
            .then(res => res.json())
            .then(data => {
                const messages = data.messages || [];
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
                            ${msg.content}
                        </div>
                        <div class="chat-timestamp">${timeString}</div>
                    `;
                }).join('');

                scrollToBottom();
            })
            .catch(err => console.error("Error loading conversation", err));
    };

    const setupChatView = (item) => {
        // UI Updates
        threadItems.forEach(t => t.classList.remove("is-active"));
        item.classList.add("is-active");

        activeUserId = item.getAttribute("data-other-id");
        activeChatName.textContent = item.getAttribute("data-other-name");
        activeChatAvatar.innerHTML = item.querySelector(".online-user-avatar").innerHTML;

        chatHeader.style.display = "flex";
        chatInputArea.style.display = "flex";
        messageInput.disabled = false;
        sendMessageBtn.disabled = false;

        chatBody.innerHTML = '<div style="margin:auto; color:var(--text-soft);">Loading messages...</div>';

        // Initial load
        loadConversation(activeUserId);

        // Start polling specifically for this conversation
        if (pollInterval) clearInterval(pollInterval);
        pollInterval = setInterval(() => {
            if (activeUserId) loadConversation(activeUserId);
        }, 5000);
    };

    // Attach click listeners to sidebar threads
    threadItems.forEach(item => {
        item.addEventListener("click", () => setupChatView(item));
    });

    // Handle sending
    const sendMessage = () => {
        const content = messageInput.value.trim();
        if (!content || !activeUserId) return;

        messageInput.disabled = true;
        sendMessageBtn.disabled = true;

        fetch("/api/messages/send", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ receiver_id: activeUserId, content: content })
        })
            .then(res => res.json())
            .then(data => {
                messageInput.disabled = false;
                sendMessageBtn.disabled = false;
                messageInput.value = "";
                messageInput.focus();
                if (data.success) {
                    // Immediately reload thread
                    loadConversation(activeUserId);
                }
            })
            .catch(err => {
                console.error(err);
                messageInput.disabled = false;
                sendMessageBtn.disabled = false;
            });
    };

    sendMessageBtn.addEventListener("click", sendMessage);
    messageInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") sendMessage();
    });
});
