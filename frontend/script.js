async function predictComplaint() {
    const complaintBox = document.getElementById("complaint");
    const button = document.getElementById("predictButton");
    const message = document.getElementById("message");
    const resultsSection = document.getElementById("results");
    const predictionList = document.getElementById("predictionList");

    const complaint = complaintBox.value.trim();

    // Basic input validation in the browser.
    if (complaint.split(/\s+/).filter(Boolean).length < 5) {
        message.textContent = "Please enter a complaint containing at least 5 words.";
        resultsSection.classList.add("hidden");
        return;
    }

    button.disabled = true;
    button.textContent = "Classifying...";
    message.textContent = "";
    resultsSection.classList.add("hidden");

    try {
        // Send the complaint text to the Python FastAPI backend.
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: complaint })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Something went wrong.");
        }

        predictionList.innerHTML = "";

        // Display the top three predicted categories.
        data.predictions.forEach(function(item) {
            const row = document.createElement("div");
            row.className = "prediction";

            const heading = document.createElement("div");
            heading.className = "prediction-heading";

            const category = document.createElement("strong");
            category.textContent = item.category;

            const confidence = document.createElement("span");
            confidence.textContent = item.confidence + "%";

            heading.appendChild(category);
            heading.appendChild(confidence);

            const background = document.createElement("div");
            background.className = "bar-background";

            const bar = document.createElement("div");
            bar.className = "bar";
            bar.style.width = item.confidence + "%";

            background.appendChild(bar);
            row.appendChild(heading);
            row.appendChild(background);
            predictionList.appendChild(row);
        });

        resultsSection.classList.remove("hidden");
    } catch (error) {
        message.textContent = error.message;
    } finally {
        button.disabled = false;
        button.textContent = "Classify Complaint";
    }
}
