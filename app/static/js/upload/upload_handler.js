document.addEventListener('DOMContentLoaded', function() {
    // Method switching functionality
    const methodSelect = document.getElementById('upload-method');
    const methodForms = document.querySelectorAll('.method-form');

    // Function to show selected method form
    function showSelectedMethod() {
        const selectedValue = methodSelect.value;

        // Hide all forms
        methodForms.forEach(form => {
            form.style.display = 'none';
        });

        // Show selected form
        const selectedForm = document.getElementById(selectedValue + '-form');
        if (selectedForm) {
            selectedForm.style.display = 'block';
        }
    }

    // Add change event listener
    methodSelect.addEventListener('change', showSelectedMethod);

    // Initial display
    showSelectedMethod();

    // Optional: Add form validation for the dataframe upload
    const dataframeForm = document.getElementById('dataframe-form');
    if (dataframeForm) {
        dataframeForm.querySelector('form').addEventListener('submit', function(e) {
            const dataframeFile = document.getElementById('dataframe_file');
            const metadataFile = document.getElementById('metadata_file');

            if (!dataframeFile.files.length || !metadataFile.files.length) {
                e.preventDefault();
                alert('Please select both a DataFrame and Metadata file.');
                return false;
            }

            if (!dataframeFile.files[0].name.endsWith('.csv') || !metadataFile.files[0].name.endsWith('.csv')) {
                e.preventDefault();
                alert('Both files must be in CSV format.');
                return false;
            }
        });
    }

    // Helpful tooltips for file upload fields
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.addEventListener('change', function() {
            if (this.files.length > 0) {
                // Update label with filename
                const fileName = this.files[0].name;
                const fileLabel = this.nextElementSibling || this.parentElement.querySelector('small');
                if (fileLabel) {
                    fileLabel.textContent = fileName;
                }
            }
        });
    });
});