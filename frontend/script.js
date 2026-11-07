const form = document.querySelector("#complaint-form");
const textarea = document.querySelector("#complaint");
const count = document.querySelector("#character-count");
const button = document.querySelector("#submit-button");
const message = document.querySelector("#message");
const results = document.querySelector("#results");
const predictionList = document.querySelector("#prediction-list");

textarea.addEventListener("input", () => {
  count.textContent =
    textarea.value.length.toLocaleString() + " / 10,000";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const text = textarea.value.trim();

  results.hidden = true;
  predictionList.replaceChildren();
  message.classList.remove("error");

  if (!text) {
    message.textContent = "Please enter a complaint first.";
    message.classList.add("error");
    return;
  }

  button.disabled = true;
  button.textContent = "Classifying…";
  message.textContent = "The local model is analyzing the complaint.";

  try {
    const response = await fetch("api/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }),
    });

    const responseText = await response.text();
    let data;

    try {
      data = JSON.parse(responseText);
    } catch {
      throw new Error(
        "The API returned a non-JSON error (" +
          response.status +
          "): " +
          responseText.slice(0, 300)
      );
    }

    if (!response.ok) {
      throw new Error(
        data.detail || "The request could not be completed."
      );
    }

    for (const prediction of data.predictions) {
      const item = document.createElement("li");
      item.className = "prediction";

      const issue = document.createElement("span");
      issue.className = "issue";
      issue.textContent = prediction.issue;

      const confidence = document.createElement("span");
      confidence.className = "confidence";
      confidence.textContent =
        (prediction.confidence * 100).toFixed(1) + "% confidence";

      item.append(issue, confidence);
      predictionList.append(item);
    }

    results.hidden = false;
    message.textContent = "";
  } catch (error) {
    message.textContent =
      error.message || "Could not reach the API. Is FastAPI running?";
    message.classList.add("error");
  } finally {
    button.disabled = false;
    button.textContent = "Classify complaint";
  }
});