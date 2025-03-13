document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileUpload');
    const uploadButton = document.getElementById('uploadButton');
    const uploadProgress = document.getElementById('uploadProgress');
    const progressBar = uploadProgress.querySelector('.progress-bar');
    const errorAlert = document.getElementById('errorAlert');
    const loadingContainer = document.getElementById('loadingContainer');
    const resultsContainer = document.getElementById('resultsContainer');
    const backButton = document.getElementById('backButton');
    
    // Handle drag and drop functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        uploadArea.classList.add('dragover');
    }
    
    function unhighlight() {
        uploadArea.classList.remove('dragover');
    }
    
    // Handle file drop
    uploadArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        fileInput.files = files;
        handleFileSelect();
    }
    
    // Handle file select via input
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });
    
    fileInput.addEventListener('change', handleFileSelect);
    
    function handleFileSelect() {
        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            const fileName = file.name;
            
            // Check file extension
            const fileExt = fileName.split('.').pop().toLowerCase();
            if (!['csv', 'xls', 'xlsx'].includes(fileExt)) {
                showError('Please select a CSV or Excel file.');
                return;
            }
            
            // Update upload area with file name
            const fileNameElement = document.createElement('p');
            fileNameElement.classList.add('selected-file');
            fileNameElement.innerHTML = `<i class="fas fa-file-alt"></i> ${fileName}`;
            
            // Clear previous content and add file name
            uploadArea.innerHTML = '';
            uploadArea.appendChild(fileNameElement);
            
            // Enable upload button
            uploadButton.disabled = false;
        }
    }
    
    // Handle upload button click
    uploadButton.addEventListener('click', uploadFile);
    
    function uploadFile() {
        if (fileInput.files.length === 0) {
            showError('Please select a file first.');
            return;
        }
        
        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('file', file);
        
        // Show progress bar
        uploadProgress.classList.remove('d-none');
        errorAlert.classList.add('d-none');
        uploadButton.disabled = true;
        
        // Show loading spinner
        loadingContainer.classList.remove('d-none');
        
        // Upload the file
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Something went wrong');
                });
            }
            return response.json();
        })
        .then(data => {
            // Hide loading and progress indicators
            loadingContainer.classList.add('d-none');
            uploadProgress.classList.add('d-none');
            
            // Display results
            displayResults(data);
        })
        .catch(error => {
            loadingContainer.classList.add('d-none');
            uploadProgress.classList.add('d-none');
            showError(error.message || 'An error occurred while processing your file.');
            uploadButton.disabled = false;
        });
    }
    
    function showError(message) {
        errorAlert.textContent = message;
        errorAlert.classList.remove('d-none');
    }
    
    // Back button handling
    backButton.addEventListener('click', () => {
        resultsContainer.classList.add('d-none');
        errorAlert.classList.add('d-none');
        uploadProgress.classList.add('d-none');
        
        // Reset upload area
        uploadArea.innerHTML = `
            <i class="fas fa-cloud-upload-alt upload-icon"></i>
            <p>Drag &amp; Drop your file here or click to browse</p>
            <p class="small text-muted">Supported formats: CSV, Excel</p>
        `;
        
        // Reset file input
        fileInput.value = '';
        uploadButton.disabled = true;
    });
    
    // Function to display results
    function displayResults(data) {
        // Show results container
        resultsContainer.classList.remove('d-none');
        
        // Fill in summary tab
        document.getElementById('filename').textContent = data.filename;
        document.getElementById('rowCount').textContent = data.insights.rows;
        document.getElementById('columnCount').textContent = data.insights.columns;
        document.getElementById('targetColumn').textContent = data.model_results.target_column;
        
        const problemType = data.model_results.problem_type === 'classification' ? 
            'Classification' : 'Regression';
        document.getElementById('problemType').textContent = problemType;
        
        document.getElementById('bestModelBadge').textContent = data.model_results.best_model;
        
        // Format score based on problem type
        let scoreText = '';
        if (data.model_results.problem_type === 'classification') {
            scoreText = `${(data.model_results.score * 100).toFixed(2)}% accuracy`;
        } else {
            scoreText = `RMSE: ${data.model_results.score.toFixed(4)}`;
        }
        document.getElementById('modelScore').textContent = scoreText;
        
        // Target distribution chart
        if (data.insights.visualizations && data.insights.visualizations.target_distribution) {
            document.getElementById('targetDistributionChart').src = 'data:image/png;base64,' + 
                data.insights.visualizations.target_distribution;
        }
        
        // Data insights tab
        populateDataInsights(data.insights);
        
        // Model comparison tab
        populateModelComparison(data.model_results);
        
        // Feature importance tab
        populateFeatureImportance(data.model_results);
    }
    
    function populateDataInsights(insights) {
        // Data sample table
        const sampleTableHeader = document.getElementById('sampleTableHeader');
        const sampleTableBody = document.getElementById('sampleTableBody');
        
        sampleTableHeader.innerHTML = '';
        sampleTableBody.innerHTML = '';
        
        if (insights.data_sample && insights.data_sample.length > 0) {
            // Add columns
            const columns = Object.keys(insights.data_sample[0]);
            columns.forEach(column => {
                const th = document.createElement('th');
                th.textContent = column;
                sampleTableHeader.appendChild(th);
            });
            
            // Add rows
            insights.data_sample.forEach(row => {
                const tr = document.createElement('tr');
                columns.forEach(column => {
                    const td = document.createElement('td');
                    td.textContent = row[column] !== null ? row[column] : 'N/A';
                    tr.appendChild(td);
                });
                sampleTableBody.appendChild(tr);
            });
        }
        
        // Feature lists
        const numericalList = document.getElementById('numericalFeaturesList');
        const categoricalList = document.getElementById('categoricalFeaturesList');
        
        numericalList.innerHTML = '';
        categoricalList.innerHTML = '';
        
        insights.numerical_features.forEach(feature => {
            const li = document.createElement('li');
            li.textContent = feature;
            numericalList.appendChild(li);
        });
        
        insights.categorical_features.forEach(feature => {
            const li = document.createElement('li');
            li.textContent = feature;
            categoricalList.appendChild(li);
        });
        
        // Correlation matrix
        if (insights.visualizations && insights.visualizations.correlation) {
            document.getElementById('correlationMatrix').src = 'data:image/png;base64,' + 
                insights.visualizations.correlation;
        } else {
            document.getElementById('correlationMatrix').src = '';
            document.getElementById('correlationMatrix').alt = 'No correlation data available';
        }
        
        // Missing values
        const missingValuesContainer = document.getElementById('missingValuesContainer');
        missingValuesContainer.innerHTML = '';
        
        const missingValues = insights.missing_values;
        const hasMissingValues = Object.values(missingValues).some(value => value > 0);
        
        if (hasMissingValues) {
            const table = document.createElement('table');
            table.className = 'table table-sm table-hover';
            
            const tableHeader = document.createElement('thead');
            const headerRow = document.createElement('tr');
            
            const headerCol1 = document.createElement('th');
            headerCol1.textContent = 'Column';
            headerRow.appendChild(headerCol1);
            
            const headerCol2 = document.createElement('th');
            headerCol2.textContent = 'Missing Values';
            headerRow.appendChild(headerCol2);
            
            tableHeader.appendChild(headerRow);
            table.appendChild(tableHeader);
            
            const tableBody = document.createElement('tbody');
            
            Object.entries(missingValues).forEach(([column, count]) => {
                if (count > 0) {
                    const row = document.createElement('tr');
                    
                    const col1 = document.createElement('td');
                    col1.textContent = column;
                    row.appendChild(col1);
                    
                    const col2 = document.createElement('td');
                    col2.textContent = count;
                    row.appendChild(col2);
                    
                    tableBody.appendChild(row);
                }
            });
            
            table.appendChild(tableBody);
            missingValuesContainer.appendChild(table);
        } else {
            const p = document.createElement('p');
            p.textContent = 'No missing values found.';
            missingValuesContainer.appendChild(p);
        }
    }
    
    function populateModelComparison(modelResults) {
        // Model comparison chart
        if (modelResults.model_comparison_chart) {
            document.getElementById('modelComparisonChart').src = 'data:image/png;base64,' + 
                modelResults.model_comparison_chart;
        }
        
        // Model scores table
        const modelScoresTableBody = document.getElementById('modelScoresTableBody');
        modelScoresTableBody.innerHTML = '';
        
        const allScores = modelResults.all_scores;
        const bestModel = modelResults.best_model;
        const problemType = modelResults.problem_type;
        
        Object.entries(allScores).forEach(([modelName, scores]) => {
            const row = document.createElement('tr');
            if (modelName === bestModel) {
                row.classList.add('model-highlight');
            }
            
            const nameCell = document.createElement('td');
            nameCell.textContent = modelName;
            row.appendChild(nameCell);
            
            if (Array.isArray(scores) && scores.length === 5) {
                // Add individual fold scores
                scores.forEach(score => {
                    const scoreCell = document.createElement('td');
                    scoreCell.textContent = formatScore(score, problemType);
                    row.appendChild(scoreCell);
                });
                
                // Add average score
                const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
                const avgScoreCell = document.createElement('td');
                avgScoreCell.textContent = formatScore(avgScore, problemType);
                avgScoreCell.style.fontWeight = 'bold';
                row.appendChild(avgScoreCell);
            } else {
                // Handle error case
                const errorCell = document.createElement('td');
                errorCell.colSpan = 6;
                errorCell.textContent = typeof scores === 'string' ? scores : 'Error during evaluation';
                errorCell.classList.add('text-danger');
                row.appendChild(errorCell);
            }
            
            modelScoresTableBody.appendChild(row);
        });
    }
    
    function formatScore(score, problemType) {
        if (problemType === 'classification') {
            return (score * 100).toFixed(2) + '%';
        } else {
            return score.toFixed(4);
        }
    }
    
    function populateFeatureImportance(modelResults) {
        const featureImportanceContainer = document.getElementById('featureImportanceContainer');
        const noFeatureImportance = document.getElementById('noFeatureImportance');
        
        if (modelResults.feature_importance && modelResults.feature_importance_chart) {
            featureImportanceContainer.classList.remove('d-none');
            noFeatureImportance.classList.add('d-none');
            
            // Feature importance chart
            document.getElementById('featureImportanceChart').src = 'data:image/png;base64,' + 
                modelResults.feature_importance_chart;
            
            // Feature importance table
            const tableBody = document.getElementById('featureImportanceTableBody');
            tableBody.innerHTML = '';
            
            Object.entries(modelResults.feature_importance).forEach(([feature, importance]) => {
                const row = document.createElement('tr');
                
                const featureCell = document.createElement('td');
                featureCell.textContent = feature;
                row.appendChild(featureCell);
                
                const importanceCell = document.createElement('td');
                importanceCell.textContent = importance.toFixed(4);
                row.appendChild(importanceCell);
                
                tableBody.appendChild(row);
            });
        } else {
            featureImportanceContainer.classList.add('d-none');
            noFeatureImportance.classList.remove('d-none');
        }
    }
});