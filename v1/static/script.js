document.getElementById("imageForm").addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData();
    const imageInput = document.getElementById("imageInput");
    const resultDiv = document.getElementById("result");
    const loadingDiv = document.getElementById("loading");
    const fileNameDisplay = document.getElementById("fileName");

    if (imageInput.files.length === 0) {
        alert("Please select an image file.");
        return;
    }

    formData.append("image", imageInput.files[0]);
    fileNameDisplay.textContent = imageInput.files[0].name;

    loadingDiv.classList.remove("hidden");
    resultDiv.innerHTML = "";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error("Error in response: " + response.statusText);
        }

        const data = await response.json();
        loadingDiv.classList.add("hidden");

        resultDiv.innerHTML = `
            <h2>🚀 Prediction Result</h2>
            <p><strong>🛠️ Predicted Class:</strong> ${data.predicted_class}</p>
            <p><strong>🔍 Classification:</strong> ${data.classification}</p>
        `;
    } catch (error) {
        console.error("Error:", error);
        loadingDiv.classList.add("hidden");
        resultDiv.innerHTML = "<p>❌ Error during prediction. Please try again.</p>";
    }
});

/* Show file name when selected */
document.getElementById("imageInput").addEventListener("change", function () {
    const fileName = this.files[0] ? this.files[0].name : "No file chosen";
    document.getElementById("fileName").textContent = fileName;
});
