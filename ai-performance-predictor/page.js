document.getElementById('add-course').addEventListener('click', function() {
    const courseContainer = document.getElementById('course-container');
    const courseCount = courseContainer.children.length + 1;
    const newCourseRow = document.createElement('div');
    newCourseRow.className = 'course-row';
    newCourseRow.innerHTML = `
        <label>Course ${courseCount}</label>
        <select class="grade">
            <option value="0-50">F</option>
            <option value="51-55">C</option>
            <option value="56-60">B</option>
            <option value="61-70">B+</option>
            <option value="71-80">A</option>
            <option value="81-90">A+</option>
            <option value="91-100">O</option>
        </select> 
        <label>Credits</label>
        <select class="credit">
            <option value="0">0</option>
            <option value="1">1</option>
            <option value="2">2</option>
            <option value="3">3</option>
            <option value="4">4</option>
            <option value="5">5</option>
        </select>
    `;
    courseContainer.appendChild(newCourseRow);
});

// Add event listener for the calculate button
document.getElementById('calculate').addEventListener('click', async function() {
    // Collect data from form
    const courseRows = document.querySelectorAll('.course-row');
    const courses = [];
    
    courseRows.forEach(row => {
        const gradeSelect = row.querySelector('.grade');
        const creditSelect = row.querySelector('.credit');
        
        // Add null check
        if (!gradeSelect || !creditSelect) {
            console.error('Could not find grade or credit select in row:', row);
            return;
        }
        
        courses.push({
            grade: gradeSelect.value,
            credits: parseInt(creditSelect.value)
        });
    });
    
    // Get attendance and CGPA
    const attendanceSelect = document.getElementById('attendance-select');
    const cgpaSelect = document.getElementById('cgpa-select');
    
    // Add null check
    if (!attendanceSelect || !cgpaSelect) {
        console.error('Could not find attendance or CGPA select elements');
        alert('Error: Form elements missing. Please refresh the page.');
        return;
    }
    
    const attendance = attendanceSelect.value;
    const cgpa = cgpaSelect.value;
    
    // Log data for debugging
    console.log('Sending data to backend:', {
        courses: courses,
        attendance: attendance,
        cgpa: cgpa
    });
    
    // Prepare data for submission
    const studentData = {
        courses: courses,
        attendance: attendance,
        cgpa: cgpa
    };
    
    try {
        // Show loading state
        document.getElementById('calculate').textContent = 'Processing...';
        document.getElementById('calculate').disabled = true;
        
        // Add debug logging for request
        console.log('Sending request to /predict endpoint');
        
        // Send data to backend for AI prediction
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(studentData)
        });
        
        console.log('Response status:', response.status);
        
        if (!response.ok) {
            throw new Error(`Network response was not ok: ${response.status}`);
        }
        
        const predictionResult = await response.json();
        console.log('Prediction result:', predictionResult);
        
        // Display the results
        displayResults(predictionResult);
    } catch (error) {
        console.error('Error in prediction request:', error);
        // Provide more detailed error information
        const errorMessage = document.getElementById('results-container');
        errorMessage.innerHTML = `
            <div class="error-message">
                <h3>Error Getting Prediction</h3>
                <p>There was a problem connecting to the prediction service:</p>
                <p><code>${error.message}</code></p>
                <p>Make sure the Flask server is running at the correct address.</p>
            </div>
        `;
        errorMessage.style.display = 'block';
        alert('Failed to get prediction. Please check the browser console for more details.');
    } finally {
        document.getElementById('calculate').textContent = 'My Performance';
        document.getElementById('calculate').disabled = false;
    }
});

// Function to display prediction results
function displayResults(results) {
    const resultsContainer = document.getElementById('results-container');
    
    // Create content for results
    resultsContainer.innerHTML = `
        <i><h2>Prediction Results</h2></i>
        <div class="results-content">
            <p><strong>Predicted Final Grade Range:</strong> ${results.predictedGrade}</p>
            <p><strong>Performance Category:</strong> ${results.performanceCategory}</p>
            <p><strong>Risk Level:</strong> ${results.riskLevel}</p>
            <div class="recommendation">
                <h3>Recommendations:</h3>
                <ul>
                    ${results.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                </ul>
            </div>
        </div>
    `;
    
    // Make sure the results container is visible
    resultsContainer.style.display = 'block';
}