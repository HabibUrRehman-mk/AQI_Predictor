const API_URL = "/api";
const button = document.querySelector("#predict-button");
const refreshButton = document.querySelector("#refresh-models");
const status = document.querySelector("#status");
const errorBox = document.querySelector("#error");

const AQI_RANGES = [
    { max: 50, label: "Good", color: "#58b978" },
    { max: 100, label: "Moderate", color: "#e3bd46" },
    { max: 150, label: "Unhealthy for sensitive groups", color: "#ee9651" },
    { max: 200, label: "Unhealthy", color: "#e05f58" },
    { max: 300, label: "Very unhealthy", color: "#9d6bb5" },
    { max: 500, label: "Hazardous", color: "#8f4058" },
];

function setText(selector, value) {
    const element = document.querySelector(selector);
    if (element) element.textContent = value;
}

function format(value, suffix = "") {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
    return `${Number(value).toFixed(1)}${suffix}`;
}

function aqiBand(value) {
    return AQI_RANGES.find((range) => value <= range.max) || AQI_RANGES[AQI_RANGES.length - 1];
}

function applyAqiColor(element, value) {
    if (!element || value === null || value === undefined) return;
    const band = aqiBand(Number(value));
    element.style.setProperty("--aqi-color", band.color);
}

function renderAqi(value, numberSelector, categorySelector, cardSelector) {
    const band = aqiBand(Number(value));
    setText(numberSelector, format(value));
    setText(categorySelector, band.label);
    applyAqiColor(document.querySelector(categorySelector), value);
    applyAqiColor(document.querySelector(cardSelector), value);
}

function svgElement(name, attributes = {}) {
    const element = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
    return element;
}

function smoothPath(points) {
    if (!points.length) return "";
    if (points.length === 1) return `M ${points[0][0]} ${points[0][1]}`;
    let path = `M ${points[0][0]} ${points[0][1]}`;
    for (let index = 1; index < points.length; index += 1) {
        const previous = points[index - 1];
        const current = points[index];
        const midpoint = (previous[0] + current[0]) / 2;
        path += ` C ${midpoint} ${previous[1]}, ${midpoint} ${current[1]}, ${current[0]} ${current[1]}`;
    }
    return path;
}

function renderChart(history, predictions, timestamp) {
    const chart = document.querySelector("#aqi-chart");
    if (!chart) return;
    chart.replaceChildren();
    const width = 900;
    const height = 330;
    const left = 16;
    const right = 12;
    const top = 16;
    const bottom = 30;
    const chartHeight = height - top - bottom;
    const current = new Date(timestamp);
    const points = [...history.slice(-73),
        { time: new Date(current.getTime() + 24 * 3600000).toISOString(), aqi: predictions[0] },
        { time: new Date(current.getTime() + 48 * 3600000).toISOString(), aqi: predictions[1] },
        { time: new Date(current.getTime() + 72 * 3600000).toISOString(), aqi: predictions[2] },
    ];
    if (!points.length) return;
    const maxValue = Math.max(...points.map((point) => Number(point.aqi) || 0));
    const yMax = Math.max(300, Math.ceil(maxValue / 50) * 50);
    const x = (index) => left + (index / Math.max(points.length - 1, 1)) * (width - left - right);
    const y = (value) => top + (1 - Math.min(Number(value), yMax) / yMax) * chartHeight;
    const ranges = [[0, 50, "#58b978"], [50, 100, "#e3bd46"], [100, 150, "#ee9651"], [150, 200, "#e05f58"], [200, Math.min(300, yMax), "#9d6bb5"], [300, yMax, "#8f4058"]];
    ranges.forEach(([from, to, color]) => {
        if (to <= from) return;
        chart.appendChild(svgElement("rect", { x: left, y: y(to), width: width - left - right, height: y(from) - y(to), fill: color, opacity: "0.08" }));
    });
    [50, 100, 150, 200, 300].forEach((mark) => chart.appendChild(svgElement("line", { x1: left, x2: width - right, y1: y(mark), y2: y(mark), stroke: "#c9d1c7", "stroke-dasharray": "3 5" })));
    const observed = points.slice(0, -3).map((point, index) => [x(index), y(point.aqi)]);
    const forecast = points.slice(-4).map((point, index) => [x(points.length - 4 + index), y(point.aqi)]);
    chart.appendChild(svgElement("path", { d: smoothPath(observed), fill: "none", stroke: "#236e68", "stroke-width": "3", "stroke-linecap": "round" }));
    chart.appendChild(svgElement("path", { d: smoothPath(forecast), fill: "none", stroke: "#d86d45", "stroke-width": "3", "stroke-dasharray": "7 6", "stroke-linecap": "round" }));
    chart.appendChild(svgElement("line", { x1: x(points.length - 4), x2: x(points.length - 4), y1: top, y2: height - bottom, stroke: "#d86d45", "stroke-dasharray": "3 5", opacity: "0.8" }));
    const handoffX = x(points.length - 4);
    const handoff = svgElement("text", { x: handoffX > width - 110 ? handoffX - 8 : handoffX + 8, y: top + 14, "text-anchor": handoffX > width - 110 ? "end" : "start", fill: "#d86d45", "font-size": "11", "font-weight": "700" });
    handoff.textContent = "FORECAST";
    chart.appendChild(handoff);
    points.forEach((point, index) => chart.appendChild(svgElement("circle", { cx: x(index), cy: y(point.aqi), r: index < points.length - 3 ? "2.3" : "5", fill: index < points.length - 3 ? "#236e68" : aqiBand(point.aqi).color, stroke: "#f8faf5", "stroke-width": "2" })));
    [0, Math.floor((points.length - 1) / 2), points.length - 1].forEach((index) => {
        const label = new Date(points[index].time).toLocaleDateString([], { month: "short", day: "numeric" });
        const text = svgElement("text", { x: x(index), y: height - 8, "text-anchor": index === 0 ? "start" : index === points.length - 1 ? "end" : "middle", fill: "#718078", "font-size": "12" });
        text.textContent = label;
        chart.appendChild(text);
    });
}

async function refreshModels() {
    refreshButton.disabled = true;
    try {
        const response = await fetch(`${API_URL}/admin/models/download`, { method: "POST" });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Model refresh failed");
    } finally {
        refreshButton.disabled = false;
    }
}

async function loadPrediction() {
    errorBox.textContent = "";
    button.disabled = true;
    status.textContent = "Loading live data";
    const city = document.querySelector("#city").value.trim();
    const latitude = document.querySelector("#latitude").value;
    const longitude = document.querySelector("#longitude").value;
    const query = new URLSearchParams({ city, latitude, longitude });
    try {
        const response = await fetch(`${API_URL}/predict?${query.toString()}`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Unable to load prediction");
        setText("#location-name", data.city);
        renderAqi(data.current_aqi, "#current-aqi", "#current-category", "#current-card");
        setText("#current-dial-value", Math.round(data.current_aqi));
        [[24, data.predicted_aqi_24h], [48, data.predicted_aqi_48h], [72, data.predicted_aqi_72h]].forEach(([horizon, value]) => renderAqi(value, `#aqi-${horizon}`, `#category-${horizon}`, `.forecast-card[data-horizon="${horizon}"]`));
        setText("#temperature", format(data.weather.temperature_2m, " °C"));
        setText("#humidity", format(data.weather.relative_humidity_2m, " %"));
        setText("#pressure", format(data.weather.surface_pressure, " hPa"));
        setText("#wind", format(data.weather.wind_speed_10m, " km/h"));
        setText("#pm25", format(data.weather.pm2_5));
        setText("#pm10", format(data.weather.pm10));
        setText("#no2", format(data.weather.nitrogen_dioxide));
        setText("#ozone", format(data.weather.ozone));
        setText("#updated-at", `Updated ${new Date(data.timestamp).toLocaleString()}`);
        renderChart(data.history || [], [data.predicted_aqi_24h, data.predicted_aqi_48h, data.predicted_aqi_72h], data.timestamp);
        status.textContent = "Live and connected";
    } catch (error) {
        status.textContent = "Unable to connect";
        errorBox.textContent = error.message;
    } finally {
        button.disabled = false;
    }
}

function showPage() {
    const page = ["metrics", "architecture"].includes(window.location.hash.slice(1)) ? window.location.hash.slice(1) : "dashboard";
    document.querySelectorAll(".page").forEach((element) => { element.hidden = element.dataset.page !== page; });
    document.querySelectorAll("[data-page-link]").forEach((link) => link.classList.toggle("active", link.dataset.pageLink === page));
    window.scrollTo({ top: 0, behavior: "smooth" });
}

button.addEventListener("click", loadPrediction);
refreshButton.addEventListener("click", () => refreshModels().catch((error) => { errorBox.textContent = error.message; }));
window.addEventListener("hashchange", showPage);
showPage();
loadPrediction();
