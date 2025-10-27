// app/static/js/instance/instance_viewer.js
document.addEventListener('DOMContentLoaded', function() {
    console.log('Instance viewer loaded');

    // Get the iframe element
    const iframe = document.querySelector('#graph-container iframe');

    // If iframe exists, add load event listener
    if (iframe) {
        iframe.addEventListener('load', function() {
            // Remove loading indicator if present
            const loadingElement = document.querySelector('.loading');
            if (loadingElement) {
                loadingElement.style.display = 'none';
            }
        });
    }
});