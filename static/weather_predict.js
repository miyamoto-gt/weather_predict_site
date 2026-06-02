async function predictWeather() {

    const data = {
        AvgTemp: Number(document.getElementById("AvgTemp").value) || 0,
        TotalPrecip: Number(document.getElementById("TotalPrecip").value) || 0,
        SolarHours: Number(document.getElementById("SolarHours").value) || 0,
        AvgCloud: Number(document.getElementById("AvgCloud").value) || 0,
        vapor_pressure: Number(document.getElementById("vapor_pressure").value) || 0,
        AvgWindSpeed: Number(document.getElementById("AvgWindSpeed").value) || 0,
        MinTemp: Number(document.getElementById("MinTemp").value) || 0
    };

    const value = document.querySelector(".result-value");
    const box = document.getElementById("result");

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error("API error");
        }

        const result = await response.json();
        const rainProb = Number(result.rain_probability);

        value.textContent = rainProb.toFixed(1) + "%";
        box.classList.remove("hidden");

    } catch (error) {
        console.error(error);
        value.textContent = "エラー";
        box.classList.remove("hidden");
    }
}


function closeResult() {
    document.getElementById("result").classList.add("hidden");
}