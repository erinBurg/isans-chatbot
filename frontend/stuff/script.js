document.getElementById('programFilterForm').addEventListener('submit', function(e) {
    e.preventDefault();

    const searchQuery = document.getElementById('search').value;
    const programPathway = document.getElementById('programPathways').value;
    const age = document.getElementById('age').value;
    const literacy = document.getElementById('literacy').value;
    const visaEligibility = document.getElementById('visaEligibility').value;
    const deliveryMethod = document.getElementById('deliveryMethod').value;
    const gender = document.getElementById('gender').value;

    console.log('Search Query:', searchQuery);
    console.log('Program Pathway:', programPathway);
    console.log('Age:', age);
    console.log('Literacy:', literacy);
    console.log('Visa Eligibility:', visaEligibility);
    console.log('Delivery Method:', deliveryMethod);
    console.log('Gender:', gender);
});

document.getElementById('resetBtn').addEventListener('click', function() {
    console.log('Form reset!');
});

async function handleKeyPress(event) {
    if (event.key === 'Enter') {
        const inputField = document.getElementById('user-input');
        const message = inputField.value.trim();
        if (message === '') return;
        inputField.value = '';

        const chatbox = document.getElementById('chatbox');
        chatbox.innerHTML += `<div class="message user-message">You: ${message}</div>`;
        chatbox.scrollTop = chatbox.scrollHeight;

        // Send message to backend
        try {
            const response = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: message })
            });
            const data = await response.json();
            chatbox.innerHTML += `<div class="message bot-message">Bot: ${data.response}</div>`;
            chatbox.scrollTop = chatbox.scrollHeight;
        }catch (error) {
            console.error('Error:', error);
            chatbox.inner
        }
      }   
    }




