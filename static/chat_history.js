// Chat History Management
document.addEventListener('DOMContentLoaded', () => {
  const chatHistoryModal = document.getElementById('chatHistoryModal');
  const chatHistoryList = document.getElementById('chatHistoryList');
  const chatHistoryItemTemplate = document.getElementById('chatHistoryItemTemplate');
  const closeChatHistoryBtn = document.getElementById('closeChatHistoryBtn');

  // Show chat history modal
  window.showChatHistory = () => {
    chatHistoryModal.classList.remove('hidden');
    loadChatHistory();
  };

  // Hide chat history modal
  closeChatHistoryBtn.addEventListener('click', () => {
    chatHistoryModal.classList.add('hidden');
  });

  // Load chat history
  async function loadChatHistory() {
    try {
      const response = await fetch('/chats');
      const chats = await response.json();

      // Clear existing items except template
      while (chatHistoryList.firstChild) {
        if (chatHistoryList.firstChild.id === 'chatHistoryItemTemplate') break;
        chatHistoryList.removeChild(chatHistoryList.firstChild);
      }

      // Add chat items with animation
      chats.forEach((chat, index) => {
        const chatItem = chatHistoryItemTemplate.content.cloneNode(true).querySelector('.chat-history-item');
        const title = chatItem.querySelector('h4');
        const details = chatItem.querySelector('p');
        const deleteBtn = chatItem.querySelector('.delete-chat-btn');

        title.textContent = chat.title;
        details.textContent = `${chat.message_count} messages • Last updated ${new Date(chat.updated_at).toLocaleString()}`;

        // Add animation delay based on index
        chatItem.style.opacity = '0';
        chatItem.style.transform = 'translateY(20px)';
        
        // Add click handler to load chat
        chatItem.addEventListener('click', (e) => {
          if (!e.target.closest('.delete-chat-btn')) {
            loadChat(chat.id);
            chatHistoryModal.classList.add('hidden');
          }
        });

        // Add delete handler
        deleteBtn.addEventListener('click', async (e) => {
          e.stopPropagation();
          if (confirm('Are you sure you want to delete this chat?')) {
            try {
              const response = await fetch(`/chats/${chat.id}`, {
                method: 'DELETE'
              });

              if (response.ok) {
                chatItem.style.opacity = '0';
                chatItem.style.transform = 'translateY(-20px)';
                setTimeout(() => chatItem.remove(), 300);
              } else {
                alert('Failed to delete chat');
              }
            } catch (error) {
              console.error('Error deleting chat:', error);
              alert('Failed to delete chat');
            }
          }
        });

        chatHistoryList.appendChild(chatItem);

        // Trigger animation after a delay
        setTimeout(() => {
          chatItem.style.transition = 'all 0.3s ease';
          chatItem.style.opacity = '1';
          chatItem.style.transform = 'translateY(0)';
        }, index * 100);
      });

    } catch (error) {
      console.error('Error loading chat history:', error);
      chatHistoryList.innerHTML = '<div class="text-red-400 text-center py-4">Failed to load chat history</div>';
    }
  }

  // Load specific chat
  async function loadChat(chatId) {
    try {
      const response = await fetch(`/chats/${chatId}`);
      const chatData = await response.json();
      // Handle chat loading logic here
      console.log('Loading chat:', chatData);
    } catch (error) {
      console.error('Error loading chat:', error);
      alert('Failed to load chat');
    }
  }
});