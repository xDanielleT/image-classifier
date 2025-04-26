// Global variables
let model;
let imageNetClasses;

// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const classifyBtn = document.getElementById('classifyBtn');
const imagePreview = document.getElementById('imagePreview');
const previewContainer = document.getElementById('previewContainer');
const resultsContainer = document.getElementById('resultsContainer');
const predictionsContainer = document.getElementById('predictions');
const loadingIndicator = document.getElementById('loading');

// Fetch ImageNet classes
async function fetchImageNetClasses() {
  try {
    const response = await fetch('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt');
    if (!response.ok) {
      throw new Error('Failed to fetch class names');
    }
    const text = await response.text();
    return text.split('\n').filter(name => name.trim() !== '');
  } catch (error) {
    console.error('Error fetching ImageNet classes:', error);
    // Fallback to a shortened list if fetch fails
    return CLASS_NAMES_FALLBACK;
  }
}

// Fallback class names (just in case the fetch fails)
const CLASS_NAMES_FALLBACK = [
    "background", "tench", "goldfish", "great white shark", "tiger shark", 
    // your existing class names array...
];

// Load the MobileNet model when the page loads
async function loadModel() {
    try {
        // Using TensorFlow.js's pre-trained MobileNet model
        model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
        console.log('MobileNet model loaded successfully');
    } catch (error) {
        console.error('Error loading the model:', error);
        alert('Failed to load the AI model. Please refresh and try again.');
    }
}

// Initialize the application
async function init() {
    // Event listeners
    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    classifyBtn.addEventListener('click', classifyImage);
    
    // Drag and drop functionality
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('highlight');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('highlight');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('highlight');
        if (e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            handleFileSelect({ target: fileInput });
        }
    });

    // Load model and classes in parallel
    try {
        loadingIndicator.style.display = 'block';
        const [classes] = await Promise.all([
            fetchImageNetClasses(),
            loadModel()
        ]);
        imageNetClasses = classes;
        console.log(`Loaded ${imageNetClasses.length} ImageNet classes`);
    } catch (error) {
        console.error('Error during initialization:', error);
    } finally {
        loadingIndicator.style.display = 'none';
    }
}

// Handle file selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file || !file.type.match('image.*')) {
        alert('Please select an image file');
        return;
    }

    const reader = new FileReader();
    reader.onload = function(e) {
        imagePreview.src = e.target.result;
        previewContainer.style.display = 'block';
        classifyBtn.disabled = false;
        resultsContainer.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// Preprocessing image for the model
function preprocessImage(image) {
    // Create a tensor from the image
    return tf.tidy(() => {
        // Convert image to tensor
        const imageTensor = tf.browser.fromPixels(image)
            .resizeNearestNeighbor([224, 224]) // Resize to model's expected input size
            .toFloat();
        
        // Normalize from [0, 255] to [-1, 1]
        const normalized = imageTensor.div(127.5).sub(1);
        
        // Expand dimensions to match model's input shape [1, 224, 224, 3]
        return normalized.expandDims(0);
    });
}

// Classify the image
async function classifyImage() {
    if (!model) {
        alert('Model not loaded yet. Please wait and try again.');
        return;
    }

    if (!imagePreview.src) {
        alert('Please select an image first');
        return;
    }

    try {
        // Show loading indicator
        loadingIndicator.style.display = 'block';
        classifyBtn.disabled = true;

        // Preprocess the image
        const tensor = preprocessImage(imagePreview);
        
        // Run inference
        const predictions = await model.predict(tensor).data();
        
        // Get the indices of the top 5 results
        const topPredictions = Array.from(predictions)
            .map((p, i) => {
                // Use fetched class names or fallback if needed
                const className = imageNetClasses && i < imageNetClasses.length 
                    ? imageNetClasses[i] 
                    : (CLASS_NAMES_FALLBACK[i] || `Class ${i}`);
                return {probability: p, className: className};
            })
            .sort((a, b) => b.probability - a.probability)
            .slice(0, 5);
        
        // Hide loading indicator
        loadingIndicator.style.display = 'none';
        
        // Display results
        displayResults(topPredictions);
        
        // Re-enable classify button
        classifyBtn.disabled = false;
    } catch (error) {
        loadingIndicator.style.display = 'none';
        classifyBtn.disabled = false;
        console.error('Error classifying image:', error);
        alert('An error occurred while classifying the image. Please try again.');
    }
}

// Display classification results
function displayResults(predictions) {
    // Clear previous results
    predictionsContainer.innerHTML = '';
    
    // Add new results
    predictions.forEach(prediction => {
        const predictionElement = document.createElement('div');
        predictionElement.className = 'prediction';
        
        const probability = (prediction.probability * 100).toFixed(2);
        
        predictionElement.innerHTML = `
            <span class="prediction-name">${prediction.className}</span>
            <span class="prediction-probability">${probability}%</span>
        `;
        
        predictionsContainer.appendChild(predictionElement);
    });
    
    // Show results container
    resultsContainer.style.display = 'block';
}

// Initialize the app when the page loads
window.addEventListener('DOMContentLoaded', init);