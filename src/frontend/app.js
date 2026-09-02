const API_URL = "/api";
const button = document.querySelector("#predict-button");
const refreshButton = document.querySelector("#refresh-models");
const status = document.querySelector("#status");
const errorBox = document.querySelector("#error");
const modelState = document.querySelector("#model-state");

function setText(id, value) {
    document.querySelector(id).textContent = value;
}

function format(value, suffix = "") {
    if (value === null || value === undefined) return "—";
    return `${Number(value).toFixed(1)}${suffix}`;
}

async function refreshModels() {
    try {
        modelState.textContent = "Refreshing model cache…";
        const response = await fetch(`${API_URL}/admin/models/download`, { method: "POST" });
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Model refresh failed");
        }

        modelState.textContent = `Loaded: ${data.models.join(", ")}`;
        return data;
    } catch (error) {
        modelState.textContent = "Model cache unavailable";
        errorBox.textContent = error.message;
        throw error;
    }
}

async function loadPrediction() {
    errorBox.textContent = "";
    button.disabled = true;
    status.textContent = "Loading…";

    const city = document.querySelector("#city").value.trim();
    const latitude = document.querySelector("#latitude").value;
    const longitude = document.querySelector("#longitude").value;

    const query = new URLSearchParams({ city, latitude, longitude });

    try {
        const response = await fetch(`${API_URL}/predict?${query.toString()}`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Unable to load prediction");
        }

        setText("#location-name", data.city);
        setText("#current-aqi", format(data.current_aqi));
        setText("#aqi-24", format(data.predicted_aqi_24h));
        setText("#aqi-48", format(data.predicted_aqi_48h));
        setText("#aqi-72", format(data.predicted_aqi_72h));

        setText("#temperature", format(data.weather.temperature_2m, " °C"));
        setText("#humidity", format(data.weather.relative_humidity_2m, " %"));
        setText("#pressure", format(data.weather.surface_pressure, " hPa"));
        setText("#wind", format(data.weather.wind_speed_10m, " km/h"));

        setText("#updated-at", `Updated ${new Date(data.timestamp).toLocaleString()}`);
        status.textContent = "Live";
    } catch (error) {
        status.textContent = "Error";
        errorBox.textContent = error.message;
    } finally {
        button.disabled = false;
    }
}

button.addEventListener("click", loadPrediction);
refreshButton.addEventListener("click", () => {
    refreshModels().catch(() => undefined);
});
window.addEventListener("load", async () => {
    try {
        await refreshModels();
    } catch (error) {
        // ignore and allow user to trigger manually
    }
    await loadPrediction();
});