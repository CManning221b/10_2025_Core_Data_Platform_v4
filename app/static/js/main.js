/**
 * Main JavaScript file for the Graph Visualization System
 * Contains global functionality used across multiple pages
 */
document.addEventListener('DOMContentLoaded', function() {
    // Initialize flash message dismissal
    const flashMessages = document.querySelectorAll('.alert');

    flashMessages.forEach(message => {
        // Add a timeout to automatically hide messages
        setTimeout(() => {
            if (message.parentNode) {
                message.style.opacity = '0';
                setTimeout(() => {
                    if (message.parentNode) {
                        message.style.display = 'none';
                    }
                }, 300);
            }
        }, 5000);
    });
});